"""
DiffMatch True Diffusion Matching
Complete implementation with DDIM sampling for dense correspondence
"""

import sys
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm
import h5py

# Add DiffMatch to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'DiffMatch'))

from train_settings.DiffMatch.feature_backbones.VGG_features import VGGPyramid
from train_settings.DiffMatch.improved_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults
from train_settings.DiffMatch.eval_utils import extract_raw_features, initialize_flow, local_Corr
from utils_flow.correlation_to_matches_utils import correlation_to_flow_w_argmax


class DiffMatchMatcher:
    def __init__(self, model_path, device='cuda', feature_size=64, ddim_steps=50):
        """
        Initialize DiffMatch with diffusion model
        
        Args:
            model_path: Path to trained DiffMatch model checkpoint
            device: cuda or cpu
            feature_size: Feature map size (default 64)
            ddim_steps: Number of DDIM sampling steps (default 50)
                       Note: Total diffusion steps is 1000, ddim_steps controls subsampling
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.feature_size = feature_size
        self.ddim_steps = ddim_steps
        self.radius = 4  # Local correlation radius
        
        print(f"Initializing DiffMatch on {self.device}")
        print(f"Feature size: {feature_size}, DDIM steps: {ddim_steps}")
        
        # Initialize VGG pyramid for feature extraction
        self.pyramid = VGGPyramid(train=False).to(self.device)
        self.pyramid.eval()
        
        # Create diffusion model with CORRECT total steps (1000) and respacing for DDIM
        self.model, self.diffusion = create_model_and_diffusion(
            image_size=64,
            class_cond=False,
            learn_sigma=False,
            sigma_small=False,
            num_channels=128,
            num_res_blocks=3,
            num_heads=4,
            num_heads_upsample=-1,
            attention_resolutions='16,8',
            dropout=0.0,
            diffusion_steps=1000,  # FIXED: Total diffusion steps (was 5!)
            noise_schedule='cosine',
            timestep_respacing=str(ddim_steps),  # DDIM subsampling (e.g., "50" for 50 steps)
            use_kl=False,
            predict_xstart=True,
            rescale_timesteps=True,
            rescale_learned_sigmas=True,
            use_checkpoint=False,
            use_scale_shift_norm=True,
            device=self.device,
            train_mode='stage_1',  # Match checkpoint architecture
        )
        
        # Load checkpoint
        if model_path and Path(model_path).exists():
            print(f"Loading model from {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint, strict=False)
        else:
            print(f"Warning: Model path {model_path} not found, using uninitialized model")
        
        self.model.to(self.device)
        self.model.eval()
        
        print("✓ DiffMatch initialized")
    
    def match_pair(self, img0, img1, min_conf=0.0):
        """
        Match a pair of images using DiffMatch diffusion sampling
        
        Args:
            img0: First image tensor [1, 3, H, W] normalized to [0, 1]
            img1: Second image tensor [1, 3, H, W]
            min_conf: Minimum confidence threshold
            
        Returns:
            kpts0: Keypoints in img0 [N, 2]
            kpts1: Keypoints in img1 [N, 2]
            matches: Match indices [M, 2]
            conf: Match confidence scores [M]
        """
        with torch.no_grad():
            # Ensure images are on correct device
            img0 = img0.to(self.device)
            img1 = img1.to(self.device)
            
            # Get original image dimensions (preserve separately from feature map dimensions)
            img_H, img_W = img0.shape[-2:]
            
            # Resize to 256 for feature extraction
            img0_256 = F.interpolate(img0, size=256, mode='bilinear', align_corners=False)
            img1_256 = F.interpolate(img1, size=256, mode='bilinear', align_corners=False)
            
            # Extract VGG features and compute correlation  
            raw_corr, c10, c20 = extract_raw_features(
                self.pyramid, img1, img0, img1_256, img0_256, self.feature_size
            )
            
            # raw_corr is in feature space [1, 1, feat_H, feat_W, feat_H, feat_W]
            B, C, feat_H, feat_W, H2, W2 = raw_corr.shape
            
            # Initialize flow from correlation (needs [B, feat_H, feat_W, feat_H, feat_W])
            init_flow = correlation_to_flow_w_argmax(
                raw_corr.squeeze(1),  # [1, feat_H, feat_W, feat_H, feat_W]
                output_shape=(feat_H, feat_W)
            )
            # init_flow shape: [B, feat_H, feat_W, 2] -> need [B, 2, feat_H, feat_W]
            if init_flow.shape[-1] == 2:
                init_flow = init_flow.permute(0, 3, 1, 2)  # [1, 2, feat_H, feat_W]
            init_flow = init_flow * feat_H  # Scale to feature map size
            
            # Build local correlation
            coords = initialize_flow(1, feat_H, feat_W, self.device)
            coords_warped = coords + init_flow
            
            # Compute local correlation (raw_corr is already [1, 1, feat_H, feat_W, feat_H, feat_W])
            local_corr = local_Corr(raw_corr, coords_warped, self.radius)
            
            # Reshape and interpolate
            local_corr = F.interpolate(
                local_corr.view(1, (2 * self.radius + 1) ** 2, feat_H, feat_W),
                size=feat_H,
                mode='bilinear',
                align_corners=True,
            )
            
            init_flow = F.interpolate(init_flow, size=feat_H, mode='bilinear', align_corners=True)
            init_flow = init_flow / feat_H  # Normalize to [-1, 1]
            
            # Prepare model inputs (stage_1 model doesn't use trg_feat)
            model_kwargs = {
                'y': None,
                'local_corr': local_corr,
                'init_flow': init_flow,
            }
            
            # Run DDIM sampling
            sample, _ = self.diffusion.ddim_sample_loop(
                self.model,
                (1, 2, self.feature_size, self.feature_size),
                noise=None,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                eta=0.0,
                progress=False,
                denoised_fn=None,
                sampling_kwargs={'src_img': img0, 'trg_img': img1},
                logger=None,
                n_batch=1,
            )
            
            # Clamp and rescale flow to downscaled image size
            sample = torch.clamp(sample, min=-1, max=1)
            flow = F.interpolate(sample, size=(img_H, img_W), mode='bilinear', align_corners=True)
            flow[:, 0] *= img_W
            flow[:, 1] *= img_H
            
            # Convert dense flow to sparse matches
            # ULTRA-PERMISSIVE: Maximum density for SfM
            # DiffMatch excels at large-baseline matching - need dense coverage
            min_conf_threshold = 0.05  # Ultra-low for maximum matches (was 0.08)
            kpts0, kpts1, matches, conf = self._flow_to_matches(flow, img_H, img_W, min_conf_threshold)
            
            return kpts0, kpts1, matches, conf
    
    def _flow_to_matches(self, flow, H, W, min_conf=0.0):
        """
        Convert dense optical flow to sparse matches with confidence
        Improved: Better confidence estimation for geometric quality
        
        Args:
            flow: Dense flow [1, 2, H, W]
            H, W: Image dimensions
            min_conf: Minimum confidence threshold
            
        Returns:
            kpts0, kpts1, matches, conf
        """
        # Sample on a grid with stride for dense matching
        # CRITICAL: Lower stride = more features = more 3D points in COLMAP
        stride = 2  # Maximum density for SfM (4x more than stride=4)
        ys = torch.arange(0, H, stride, device=flow.device)
        xs = torch.arange(0, W, stride, device=flow.device)
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        
        # Source keypoints
        kpts0 = torch.stack([xx.flatten(), yy.flatten()], dim=1).float()  # [N, 2]
        
        # Target keypoints from flow
        flow_sampled = flow[0, :, yy.long(), xx.long()]  # [2, Ny, Nx]
        flow_sampled = flow_sampled.permute(1, 2, 0).reshape(-1, 2)  # [N, 2]
        kpts1 = kpts0 + flow_sampled
        
        # Filter out-of-bounds points
        valid = (kpts1[:, 0] >= 0) & (kpts1[:, 0] < W) & \
                (kpts1[:, 1] >= 0) & (kpts1[:, 1] < H)
        
        kpts0 = kpts0[valid]
        kpts1 = kpts1[valid]
        flow_sampled = flow_sampled[valid]
        
        # Improved confidence calculation
        # Factor 1: Flow magnitude (smaller = better)
        flow_mag = torch.sqrt(flow_sampled[:, 0]**2 + flow_sampled[:, 1]**2)
        conf_mag = torch.exp(-flow_mag / 50.0)  # Exponential decay, sharper cutoff
        
        # Factor 2: Flow smoothness (check local variance)
        # Reshape flow to compute local gradients
        flow_2d = flow[0]  # [2, H, W]
        # Compute flow gradients (approximation)
        flow_grad_x = torch.abs(flow_2d[:, :, 1:] - flow_2d[:, :, :-1])  # [2, H, W-1]
        flow_grad_y = torch.abs(flow_2d[:, 1:, :] - flow_2d[:, :-1, :])  # [2, H-1, W]
        # Pad to match size
        flow_grad_x = F.pad(flow_grad_x, (0, 1), mode='replicate')  # [2, H, W]
        flow_grad_y = F.pad(flow_grad_y, (0, 0, 0, 1), mode='replicate')  # [2, H, W]
        flow_smoothness = torch.sqrt(flow_grad_x**2 + flow_grad_y**2).sum(dim=0)  # [H, W]
        
        # Sample smoothness at keypoint locations
        smoothness_sampled = flow_smoothness[yy.long(), xx.long()].flatten()[valid]
        conf_smooth = torch.exp(-smoothness_sampled / 10.0)  # Lower gradient = smoother = better
        
        # Combined confidence: geometric mean
        conf = torch.sqrt(conf_mag * conf_smooth)
        
        # Filter by confidence threshold
        if min_conf > 0:
            conf_mask = conf >= min_conf
            kpts0 = kpts0[conf_mask]
            kpts1 = kpts1[conf_mask]
            conf = conf[conf_mask]
        
        # Matches are simply indices
        matches = torch.arange(len(kpts0), device=flow.device).unsqueeze(1)
        matches = torch.cat([matches, matches], dim=1)  # [M, 2]
        
        return kpts0.cpu().numpy(), kpts1.cpu().numpy(), matches.cpu().numpy(), conf.cpu().numpy()


def match_pairs_with_diffmatch(
    image_dir,
    pairs_file,
    features_output,
    matches_output,
    model_path,
    feature_size=64,
    ddim_steps=50,
    device='cuda',
    max_image_size=1024
):
    """
    Match image pairs using full DiffMatch diffusion inference
    
    Args:
        image_dir: Directory containing images
        pairs_file: Text file with image pairs
        features_output: Output h5 file for features
        matches_output: Output h5 file for matches
        model_path: Path to DiffMatch checkpoint
        feature_size: Feature map resolution
        ddim_steps: Number of DDIM sampling steps
        device: Device to use
        max_image_size: Max image dimension, will resize preserving aspect ratio
        max_image_size: Maximum dimension for images (to save VRAM)
    """
    from PIL import Image
    import torchvision.transforms as transforms
    
    image_dir = Path(image_dir)
    
    # Read pairs
    with open(pairs_file, 'r') as f:
        pairs = [line.strip().split() for line in f]
    
    # Initialize matcher
    matcher = DiffMatchMatcher(model_path, device, feature_size, ddim_steps)
    
    # Image transform with 4x downscale (optimal balance: features vs precision)
    def load_and_downscale_image(img_path, scale_factor=4):
        """Load image and downscale by scale_factor"""
        img = Image.open(img_path).convert('RGB')
        orig_w, orig_h = img.size
        new_w = orig_w // scale_factor
        new_h = orig_h // scale_factor
        img_resized = img.resize((new_w, new_h), Image.LANCZOS)
        # CRITICAL: Return actual scale ratios, not uniform scale_factor
        # This handles cases where orig dimensions aren't perfectly divisible
        scale_x = orig_w / new_w
        scale_y = orig_h / new_h
        return img_resized, (orig_w, orig_h), (scale_x, scale_y)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    print(f"\nMatching {len(pairs)} image pairs with DiffMatch diffusion...")
    print(f"DDIM steps per pair: {ddim_steps}")
    print(f"Estimated time: ~{len(pairs) * ddim_steps * 0.15:.0f}s ({len(pairs) * ddim_steps * 0.15 / 60:.1f} min)")
    print("="*80)
    
    # Store features and matches
    features_dict = {}
    matches_dict = {}
    
    # First pass: Extract all keypoints for each image from all pairs
    image_keypoints = {}  # {img_name: list of (kpts, pair_key, is_img0)}
    
    import time
    start_time = time.time()
    
    for idx, (img0_name, img1_name) in enumerate(tqdm(pairs, desc="🔄 DiffMatch matching", unit="pair")):
        img0_path = image_dir / img0_name
        img1_path = image_dir / img1_name
        
        if not img0_path.exists() or not img1_path.exists():
            continue
        
        # Load and downscale images (4x smaller: optimal spatial hashing)
        img0_pil, orig_size0, (scale0_x, scale0_y) = load_and_downscale_image(img0_path)
        img1_pil, orig_size1, (scale1_x, scale1_y) = load_and_downscale_image(img1_path)
        
        img0 = transform(img0_pil).unsqueeze(0)
        img1 = transform(img1_pil).unsqueeze(0)
        
        # Match
        try:
            pair_start = time.time()
            kpts0, kpts1, matches, conf = matcher.match_pair(img0, img1)
            pair_time = time.time() - pair_start
            
            # Scale keypoints from downscaled image space to original image size
            # kpts0/kpts1 are currently in downscaled image coordinates (img_H × img_W)
            # Need to scale to original image coordinates (orig_size0 × orig_size1)
            # CRITICAL: Use separate X/Y scales to handle non-uniform downscaling
            kpts0_orig = kpts0.copy()
            kpts0_orig[:, 0] *= scale0_x  # X coordinate uses X scale
            kpts0_orig[:, 1] *= scale0_y  # Y coordinate uses Y scale
            
            kpts1_orig = kpts1.copy()
            kpts1_orig[:, 0] *= scale1_x
            kpts1_orig[:, 1] *= scale1_y
            
            # Clip keypoints to valid image boundaries
            kpts0_orig[:, 0] = np.clip(kpts0_orig[:, 0], 0, orig_size0[0] - 1)
            kpts0_orig[:, 1] = np.clip(kpts0_orig[:, 1], 0, orig_size0[1] - 1)
            kpts1_orig[:, 0] = np.clip(kpts1_orig[:, 0], 0, orig_size1[0] - 1)
            kpts1_orig[:, 1] = np.clip(kpts1_orig[:, 1], 0, orig_size1[1] - 1)
            
            # Calculate ETA
            elapsed = time.time() - start_time
            avg_time = elapsed / (idx + 1)
            remaining = (len(pairs) - idx - 1) * avg_time
            
            # Update progress bar with stats
            tqdm.write(f"  [{idx+1}/{len(pairs)}] {img0_name} ↔ {img1_name}: "
                      f"{len(matches)} matches in {pair_time:.1f}s | "
                      f"Elapsed: {elapsed/60:.1f}min | ETA: {remaining/60:.1f}min")
            
            # Collect keypoints from all pairs
            pair_key = f"{img0_name}_{img1_name}"
            
            if img0_name not in image_keypoints:
                image_keypoints[img0_name] = []
            image_keypoints[img0_name].append((kpts0_orig, pair_key, True, matches[:, 0]))
            
            if img1_name not in image_keypoints:
                image_keypoints[img1_name] = []
            image_keypoints[img1_name].append((kpts1_orig, pair_key, False, matches[:, 1]))
            
            # Store raw matches for later index remapping
            matches_dict[pair_key] = {
                'matches0': matches[:, 0],  # indices in img0 keypoints  
                'matches1': matches[:, 1],  # indices in img1 keypoints
                'scores': conf
            }
            
        except Exception as e:
            tqdm.write(f"  ✗ Failed {img0_name} - {img1_name}: {e}")
            continue
    
    total_time = time.time() - start_time
    print("="*80)
    print(f"✓ Matching complete in {total_time/60:.1f} minutes")
    print(f"  Average: {total_time/len(pairs):.1f}s per pair")
    print(f"  Success rate: {len(matches_dict)}/{len(pairs)} pairs")
    print("="*80)
    
    # Consolidate keypoints for each image and build index mapping
    # CRITICAL: Use spatial hashing to merge duplicate features from different pairs
    # This enables COLMAP to build feature tracks across multiple images
    print(f"\nConsolidating features and merging duplicates...")
    index_mapping = {}  # {img_name: {pair_key: {local_idx: global_idx}}}
    
    # Spatial Hash Map: {img_name: {(x_int, y_int): global_idx}}
    # Used to detect and merge features at the same spatial location
    spatial_lookup = {}
    merge_stats = {'total_features': 0, 'merged_features': 0, 'unique_features': 0}
    
    for img_name, kpt_list in tqdm(image_keypoints.items(), desc="Consolidating images", unit="img"):
        all_kpts = []
        spatial_lookup[img_name] = {}
        
        # Initialize mapping
        if img_name not in index_mapping:
            index_mapping[img_name] = {}
        
        for kpts, pair_key, is_img0, local_indices in kpt_list:
            if pair_key not in index_mapping[img_name]:
                index_mapping[img_name][pair_key] = {}
            
            for i, local_idx in enumerate(local_indices):
                # Get keypoint coordinates
                x, y = kpts[i]
                merge_stats['total_features'] += 1
                
                # Spatial Quantization: round to integer pixel coordinates
                # This allows us to merge duplicate features from different pairs
                # that are at the same or very close locations
                coord_key = (int(round(x)), int(round(y)))
                
                if coord_key in spatial_lookup[img_name]:
                    # Feature already exists at this location - reuse existing global ID
                    # This creates feature tracks: B <-> A <-> C
                    global_idx = spatial_lookup[img_name][coord_key]
                    merge_stats['merged_features'] += 1
                else:
                    # New feature location - assign new global ID
                    global_idx = len(all_kpts)
                    all_kpts.append(kpts[i])
                    spatial_lookup[img_name][coord_key] = global_idx
                    merge_stats['unique_features'] += 1
                
                # Record local-to-global mapping for this pair
                index_mapping[img_name][pair_key][int(local_idx)] = global_idx
        
        # Convert to numpy array and store
        if len(all_kpts) > 0:
            stacked_kpts = np.array(all_kpts)
            # Create dummy descriptors (DiffMatch has no descriptors)
            descriptors = np.zeros((len(stacked_kpts), 128), dtype=np.float32)
            
            features_dict[img_name] = {
                'keypoints': stacked_kpts,
                'descriptors': descriptors
            }
        else:
            print(f"  Warning: {img_name} has no keypoints")
    
    # Print spatial hashing effectiveness
    merge_rate = (merge_stats['merged_features'] / merge_stats['total_features'] * 100) if merge_stats['total_features'] > 0 else 0
    print(f"\n📊 Spatial Hashing Stats:")
    print(f"  Total features before merge: {merge_stats['total_features']}")
    print(f"  Unique features after merge: {merge_stats['unique_features']}")
    print(f"  Duplicates merged: {merge_stats['merged_features']} ({merge_rate:.1f}%)")
    print(f"  Compression ratio: {merge_stats['total_features'] / max(1, merge_stats['unique_features']):.2f}x")
    
    # Update match indices using the index mapping
    print(f"\nRemapping match indices...")
    final_matches_dict = {}
    
    for pair_key in matches_dict:
        # pair_key format: "img0_img1"
        parts = pair_key.split('_')
        # Find the split point (after first image extension)
        img0_name = None
        img1_name = None
        for i, part in enumerate(parts):
            if part.endswith('.JPG') or part.endswith('.jpg'):
                img0_name = '_'.join(parts[:i+1])
                img1_name = '_'.join(parts[i+1:])
                break
        
        if img0_name not in features_dict or img1_name not in features_dict:
            continue
            
        # Get total number of keypoints in img0 (this is CRITICAL for hloc format)
        total_kpts0 = len(features_dict[img0_name]['keypoints'])
        
        # Initialize DENSE arrays with -1 (no match) and 0.0 (no score) for ALL img0 keypoints
        # Both arrays must have the same length = total_kpts0
        dense_matches0 = np.full(total_kpts0, -1, dtype=np.int32)
        dense_scores = np.zeros(total_kpts0, dtype=np.float32)
        
        old_matches0 = matches_dict[pair_key]['matches0']
        old_matches1 = matches_dict[pair_key]['matches1']
        scores = matches_dict[pair_key]['scores']
        
        # Map old local indices to new global indices
        for i in range(len(old_matches0)):
            local_idx0 = int(old_matches0[i])
            local_idx1 = int(old_matches1[i])
            
            # Check if mapping exists
            if (pair_key in index_mapping[img0_name] and 
                local_idx0 in index_mapping[img0_name][pair_key] and
                pair_key in index_mapping[img1_name] and 
                local_idx1 in index_mapping[img1_name][pair_key]):
                
                global_idx0 = index_mapping[img0_name][pair_key][local_idx0]
                global_idx1 = index_mapping[img1_name][pair_key][local_idx1]
                
                # Fill in the match: img0's keypoint at global_idx0 matches img1's keypoint at global_idx1
                dense_matches0[global_idx0] = global_idx1
                # Fill in the corresponding score
                dense_scores[global_idx0] = scores[i]
        
        final_matches_dict[pair_key] = {
            'matches0': dense_matches0,  # DENSE array: length = total_kpts0
            'scores': dense_scores        # DENSE array: length = total_kpts0
        }
    
    
    # Save features
    print(f"Saving features to {features_output}")
    with h5py.File(features_output, 'w') as f:
        for img_name, feat in tqdm(features_dict.items(), desc="Saving features", unit="img"):
            grp = f.create_group(img_name)
            grp.create_dataset('keypoints', data=feat['keypoints'])
            grp.create_dataset('descriptors', data=feat['descriptors'])
    
    # Save matches
    print(f"Saving matches to {matches_output}")
    with h5py.File(matches_output, 'w') as f:
        for pair_key, match in tqdm(final_matches_dict.items(), desc="Saving matches", unit="pair"):
            grp = f.create_group(pair_key)
            grp.create_dataset('matches0', data=match['matches0'])
            grp.create_dataset('matching_scores0', data=match['scores'])
    
    print(f"✓ Saved {len(features_dict)} features and {len(final_matches_dict)} matches")
    return features_output, matches_output


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--pairs_file', type=str, required=True)
    parser.add_argument('--features_output', type=str, required=True)
    parser.add_argument('--matches_output', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--feature_size', type=int, default=64)
    parser.add_argument('--ddim_steps', type=int, default=50)
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    match_pairs_with_diffmatch(
        args.image_dir,
        args.pairs_file,
        args.features_output,
        args.matches_output,
        args.model_path,
        args.feature_size,
        args.ddim_steps,
        args.device
    )
