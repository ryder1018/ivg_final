#!/usr/bin/env python3
"""
Complete DiffMatch + hloc + NeRF Pipeline for Dense 3D Reconstruction

Pipeline:
1. Generate exhaustive pairs
2. DiffMatch: Feature extraction and diffusion matching (DDIM sampling)
3. hloc: Structure-from-Motion (camera poses + sparse point cloud)
4. NeRF: Dense depth maps + dense 3D point cloud
"""

import sys
from pathlib import Path
import subprocess
import pycolmap
import numpy as np
from tqdm import tqdm
import h5py
import sqlite3

# Import hloc
from hloc import pairs_from_exhaustive, reconstruction


def run_diffmatch_pipeline(
    image_dir: Path,
    output_dir: Path,
    model_path: Path,
    feature_size: int = 64,
    ddim_steps: int = 50,
    max_iterations: int = 30000,
):
    """Complete DiffMatch reconstruction pipeline"""
    
    print("="*80)
    print("DiffMatch + hloc + NeRF Pipeline for Dense 3D Reconstruction")
    print("="*80)
    print(f"Image directory: {image_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Model: {model_path}")
    print(f"DDIM steps: {ddim_steps} (subsampling from 1000 total diffusion steps)")
    print("="*80)
    
    # Create output structure
    output_dir.mkdir(parents=True, exist_ok=True)
    features_path = output_dir / "features.h5"
    matches_path = output_dir / "matches.h5"
    pairs_path = output_dir / "pairs.txt"
    sfm_dir = output_dir / "sparse"
    sfm_dir.mkdir(exist_ok=True)
    
    # Get image list
    image_list = sorted([
        p.relative_to(image_dir).as_posix() 
        for p in image_dir.iterdir() 
        if p.suffix.lower() in ['.jpg', '.jpeg', '.png']
    ])
    
    print(f"\n[1/5] Found {len(image_list)} images")
    
    # Step 1: Generate image pairs (exhaustive for dense reconstruction)
    print(f"\n[2/5] Generating exhaustive image pairs...")
    pairs_from_exhaustive.main(pairs_path, image_list=image_list)
    
    with open(pairs_path, 'r') as f:
        num_pairs = sum(1 for _ in f)
    print(f"    Generated {num_pairs} pairs")
    
    # Step 2: DiffMatch feature extraction and matching with diffusion
    print(f"\n[3/5] Running DiffMatch diffusion matching...")
    print(f"    Pairs: {num_pairs}")
    print(f"    DDIM steps per pair: {ddim_steps}")
    print(f"    Estimated time: ~{num_pairs * ddim_steps * 0.15 / 60:.1f} minutes")
    
    from diffmatch_matcher import match_pairs_with_diffmatch
    
    import time
    start_time = time.time()
    
    match_pairs_with_diffmatch(
        image_dir=image_dir,
        pairs_file=pairs_path,
        features_output=features_path,
        matches_output=matches_path,
        model_path=str(model_path),
        feature_size=feature_size,
        ddim_steps=ddim_steps,
        device='cuda',
        max_image_size=1024
    )
    
    elapsed = time.time() - start_time
    print(f"\n✓ DiffMatch matching completed in {elapsed/60:.1f} minutes")
    
    # Step 3: Run COLMAP reconstruction using hloc
    print(f"\n[4/5] Running COLMAP reconstruction with hloc...")
    print(f"  Note: This may take several minutes due to large keypoint count")
    print(f"  Expected stages:")
    print(f"    1. Import images")
    print(f"    2. Import features (~47 images)")
    print(f"    3. Import matches (~1081 pairs)")
    print(f"    4. Geometric verification")
    print(f"    5. Incremental reconstruction")
    print()
    
    # Use hloc's reconstruction.main() with default parameters
    # (bypassing custom mapper options due to API incompatibility)
    print(f"  Running hloc reconstruction with default COLMAP parameters...")
    model = reconstruction.main(
        sfm_dir=sfm_dir,
        image_dir=image_dir,
        pairs=pairs_path,
        features=features_path,
        matches=matches_path,
        camera_mode=pycolmap.CameraMode.PER_IMAGE,
        verbose=True,
        image_options={"camera_model": "SIMPLE_RADIAL"},
    )
    
    # Print statistics
    print("\n" + "="*80)
    print("DiffMatch + COLMAP Results")
    print("="*80)
    
    if model is None:
        print("✗ COLMAP reconstruction FAILED")
        print("  Possible reasons:")
        print("  1. Not enough inlier matches after geometric verification")
        print("  2. Insufficient 3D points for triangulation")
        print("  3. Poor initial image pair quality")
        print("\nTroubleshooting:")
        print("  - Check if matches.h5 has enough correspondences")
        print("  - Try adjusting stride in _flow_to_matches (currently stride=8)")
        print("  - Verify match quality and distribution")
        print("="*80)
        return None
    
    print(f"✓ Registered images: {model.num_images()}/{len(image_list)}")
    print(f"✓ 3D points: {model.num_points3D()}")
    print(f"✓ Mean reprojection error: {model.compute_mean_reprojection_error():.2f} pixels")
    print("="*80)
    
    # Export COLMAP results
    export_colmap_results(model, output_dir)
    
    # Step 4: Train NeRF for dense reconstruction
    print("\n[5/5] Training NeRF for Dense Reconstruction")
    print("="*80)
    nerf_output = train_nerf(
        image_dir=image_dir,
        colmap_dir=sfm_dir,
        output_dir=output_dir,
        max_iterations=max_iterations,
    )
    
    if nerf_output:
        export_nerf_outputs(nerf_output, output_dir)
    
    return model


def export_colmap_results(model, output_dir: Path):
    """Export COLMAP reconstruction results"""
    import open3d as o3d
    
    print("\nExporting COLMAP results...")
    
    # Export point cloud
    points = []
    colors = []
    
    for point3D in model.points3D.values():
        points.append(point3D.xyz)
        colors.append(point3D.color / 255.0)
    
    if len(points) == 0:
        print("Warning: No 3D points to export")
        return
    
    points = np.array(points)
    colors = np.array(colors)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    ply_path = output_dir / "diffmatch_colmap_sparse.ply"
    o3d.io.write_point_cloud(str(ply_path), pcd)
    print(f"✓ COLMAP sparse point cloud: {ply_path} ({len(points)} points)")
    
    # Export camera poses
    poses_path = output_dir / "camera_poses.txt"
    with open(poses_path, 'w') as f:
        f.write("# image_id image_name qw qx qy qz tx ty tz\n")
        for image in model.images.values():
            pose = image.cam_from_world()
            quat = pose.rotation.quat
            trans = pose.translation
            f.write(f"{image.image_id} {image.name} ")
            f.write(f"{quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f} ")
            f.write(f"{trans[0]:.6f} {trans[1]:.6f} {trans[2]:.6f}\n")
    
    print(f"✓ Camera poses: {poses_path}")


def train_nerf(image_dir: Path, colmap_dir: Path, output_dir: Path, max_iterations: int):
    """Train NeRF using nerfstudio"""
    import shutil
    
    nerf_output = output_dir / "nerf"
    
    # Prepare data directory
    data_dir = output_dir / "nerfstudio_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n[1/2] Preparing data for NeRF...")
    
    # Copy images
    images_dest = data_dir / "images"
    if images_dest.exists():
        shutil.rmtree(images_dest)
    shutil.copytree(image_dir, images_dest)
    
    # Copy COLMAP results
    colmap_dest = data_dir / "colmap" / "sparse" / "0"
    colmap_dest.parent.parent.mkdir(parents=True, exist_ok=True)
    if colmap_dest.exists():
        shutil.rmtree(colmap_dest)
    
    # Find COLMAP bin files
    if list(colmap_dir.glob("*.bin")):
        actual_colmap_dir = colmap_dir
    elif (colmap_dir / "0").exists() and list((colmap_dir / "0").glob("*.bin")):
        actual_colmap_dir = colmap_dir / "0"
    elif (colmap_dir / "models" / "0").exists():
        actual_colmap_dir = colmap_dir / "models" / "0"
    else:
        raise FileNotFoundError(f"Cannot find COLMAP bin files in {colmap_dir}")
    
    print(f"   Copying COLMAP from: {actual_colmap_dir}")
    colmap_dest.mkdir(parents=True, exist_ok=True)
    
    for bin_file in ['cameras.bin', 'images.bin', 'points3D.bin']:
        src = actual_colmap_dir / bin_file
        if src.exists():
            shutil.copy2(src, colmap_dest / bin_file)
    
    print("   ✓ Data prepared")
    
    # Pre-downscale images to avoid interactive prompt
    print("   Pre-downscaling images...")
    downscale_cmd = [
        "ns-process-data", "images",
        "--data", str(data_dir),
        "--skip-colmap",
        "--downscale-factor", "4",
    ]
    try:
        subprocess.run(downscale_cmd, check=True, capture_output=True, text=True)
        print("   ✓ Images downscaled")
    except subprocess.CalledProcessError as e:
        print(f"   Note: Downscale may have already been done ({e})")
    
    # Train NeRF
    print(f"\n[2/2] Training NeRF ({max_iterations} iterations)...")
    print("="*80)
    cmd = [
        "ns-train", "nerfacto",
        "--max-num-iterations", str(max_iterations),
        "--output-dir", str(nerf_output),
        "colmap",
        "--data", str(data_dir),
        "--auto-scale-poses", "True",
        "--downscale-factor", "4",
    ]
    
    print(f"Running: {' '.join(cmd)}\n")
    try:
        subprocess.run(cmd, check=True)
        print(f"\n✓ NeRF training complete: {nerf_output}")
        return nerf_output
    except subprocess.CalledProcessError as e:
        print(f"\n✗ NeRF training failed: {e}")
        return None


def export_nerf_outputs(nerf_output: Path, output_dir: Path):
    """Export NeRF dense point cloud and depth maps"""
    print("\n" + "="*80)
    print("Exporting NeRF Dense Reconstruction")
    print("="*80)
    
    # Find trained model
    model_dirs = list(nerf_output.glob("*/nerfacto/*/"))
    if not model_dirs:
        model_dirs = list(nerf_output.glob("nerfacto/*/"))
    if not model_dirs:
        print("Warning: No trained NeRF model found")
        return
    
    latest_model = sorted(model_dirs)[-1]
    config_path = latest_model / "config.yml"
    
    # Export dense point cloud (1M points)
    print("\n[1/2] Exporting dense point cloud (1M points)...")
    point_cloud_path = output_dir / "diffmatch_nerf_dense.ply"
    cmd = [
        "ns-export", "pointcloud",
        "--load-config", str(config_path),
        "--output-dir", str(output_dir),
        "--num-points", "1000000",
        "--remove-outliers", "True",
        "--use-bounding-box", "True",
    ]
    try:
        subprocess.run(cmd, check=True)
        # Rename output
        default_output = output_dir / "point_cloud.ply"
        if default_output.exists():
            default_output.rename(point_cloud_path)
        print(f"✓ Dense point cloud: {point_cloud_path}")
    except subprocess.CalledProcessError as e:
        print(f"✗ Point cloud export failed: {e}")
    
    # Export depth maps
    print("\n[2/2] Exporting depth maps...")
    depth_output = output_dir / "depth_maps"
    depth_output.mkdir(exist_ok=True)
    cmd = [
        "ns-render", "dataset",
        "--load-config", str(config_path),
        "--output-path", str(depth_output),
        "--rendered-output-names", "depth",
    ]
    try:
        subprocess.run(cmd, check=True)
        print(f"✓ Depth maps: {depth_output}")
    except subprocess.CalledProcessError as e:
        print(f"✗ Depth map export failed: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="DiffMatch + hloc + NeRF Pipeline"
    )
    parser.add_argument(
        "--image_dir",
        type=Path,
        required=True,
        help="Directory containing input images"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Output directory"
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        default="/home/c0922/IVG/final-project/DiffMatch/model_best_dped.pt",
        help="DiffMatch model checkpoint"
    )
    parser.add_argument(
        "--feature_size",
        type=int,
        default=64,
        help="Feature map resolution"
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="Number of DDIM sampling steps (default: 50, more steps = better quality but slower)"
    )
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=30000,
        help="Max NeRF training iterations"
    )
    
    args = parser.parse_args()
    # Run pipeline
    model = run_diffmatch_pipeline(
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        model_path=args.model_path,
        feature_size=args.feature_size,
        ddim_steps=args.ddim_steps,
        max_iterations=args.max_iterations,
    )
    
    # Print final summary
    print("\n" + "="*80)
    print("✓ DiffMatch Pipeline Completed!")
    print("="*80)
    print(f"\nResults in: {args.output_dir}")
    print(f"  - COLMAP sparse: {args.output_dir / 'diffmatch_colmap_sparse.ply'}")
    print(f"  - Camera poses: {args.output_dir / 'camera_poses.txt'}")
    print(f"  - NeRF dense (1M): {args.output_dir / 'diffmatch_nerf_dense.ply'}")
    print(f"  - Depth maps: {args.output_dir / 'depth_maps'}")
    print("="*80)
