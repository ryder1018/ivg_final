#!/usr/bin/env python3
"""
LightGlue + hloc + NeRF Pipeline for Dense 3D Reconstruction
Based on official LightGlue recommendation (using hloc toolbox)

Pipeline:
1. LightGlue: Feature extraction and matching
2. hloc: Structure-from-Motion (camera poses + sparse point cloud)
3. NeRF: Dense depth maps + dense 3D point cloud
"""

import sys
from pathlib import Path
import argparse
import subprocess
import pycolmap
from tqdm import tqdm

# Import from LightGlue repo
sys.path.insert(0, str(Path(__file__).parent.parent / "LightGlue"))

# Import hloc (official tool recommended by LightGlue for SfM)
from hloc import (
    extract_features,
    match_features,
    pairs_from_exhaustive,
    reconstruction,
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def _is_image_file(path: Path) -> bool:
    return path.suffix.casefold() in IMAGE_EXTS


def lightglue_reconstruction_pipeline(
    image_dir: Path,
    output_dir: Path,
    feature_type: str = "disk",  # disk, aliked-n16, superpoint_max
    matcher_type: str = "disk+lightglue",  # disk+lightglue, aliked+lightglue
    num_matched: int = None,
    skip_nerf: bool = False,
    max_iterations: int = 30000,
    enable_rich: bool = False,
):
    """
    Complete LightGlue + hloc + NeRF reconstruction pipeline
    
    Steps:
    1. LightGlue: Extract features and match
    2. hloc: Structure-from-Motion (camera poses)
    3. NeRF: Dense depth maps and 3D point cloud
    """
    
    print("="*70)
    print("LightGlue Official 3D Reconstruction Pipeline")
    print("="*70)
    print(f"Image directory: {image_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Feature type: {feature_type}")
    print(f"Matcher type: {matcher_type}")
    print("="*70)
    
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
        if _is_image_file(p)
    ])
    
    print(f"\n[1/4] Found {len(image_list)} images")
    
    # Step 1: Extract features using LightGlue-compatible extractors
    print(f"\n[2/4] Extracting features with {feature_type}...")
    feature_conf = extract_features.confs[feature_type]
    extract_features.main(
        conf=feature_conf,
        image_dir=image_dir,
        image_list=image_list,
        feature_path=features_path,
    )
    
    # Step 2: Generate image pairs (exhaustive for dense reconstruction)
    print(f"\n[3/4] Generating image pairs...")
    if num_matched is None:
        pairs_from_exhaustive.main(pairs_path, image_list=image_list)
    else:
        print(f"    Using sequential pairing: {num_matched} neighbors per image")
        with open(pairs_path, "w") as f:
            for i, name in enumerate(image_list):
                for j in range(1, num_matched + 1):
                    if i + j < len(image_list):
                        f.write(f"{name} {image_list[i + j]}\n")
    
    # Count pairs
    with open(pairs_path, 'r') as f:
        num_pairs = sum(1 for _ in f)
    print(f"    Generated {num_pairs} pairs")
    
    # Step 3: Match features using LightGlue
    print(f"\n[4/4] Matching features with {matcher_type}...")
    matcher_conf = match_features.confs[matcher_type]
    match_features.main(
        conf=matcher_conf,
        pairs=pairs_path,
        features=features_path,
        matches=matches_path,
    )
    
    # Step 4: Run COLMAP reconstruction
    print(f"\n[5/5] Running COLMAP reconstruction...")
    model = reconstruction.main(
        sfm_dir=sfm_dir,
        image_dir=image_dir,
        pairs=pairs_path,
        features=features_path,
        matches=matches_path,
        camera_mode=pycolmap.CameraMode.PER_IMAGE,  # Handle different image sizes
        verbose=True,
    )
    
    # Print statistics
    print("\n" + "="*70)
    print("Structure-from-Motion Results")
    print("="*70)
    print(f"Registered images: {model.num_images()}")
    print(f"3D points: {model.num_points3D()}")
    print(f"Mean reprojection error: {model.compute_mean_reprojection_error():.2f} pixels")
    print("="*70)
    
    # Export COLMAP point cloud
    export_results(model, output_dir)
    
    # Step 5: Train NeRF for dense reconstruction
    if not skip_nerf:
        print("\n" + "="*70)
        print("[Step 5] Training NeRF for Dense Reconstruction")
        print("="*70)
        nerf_output = train_nerf(
            image_dir=image_dir,
            colmap_dir=sfm_dir,  # Use sparse/ directly, not sparse/0
            output_dir=output_dir,
            max_iterations=max_iterations,
            enable_rich=enable_rich,
        )
        
        if nerf_output:
            export_nerf_outputs(nerf_output, output_dir)
    
    return model


def export_results(model, output_dir: Path):
    """Export COLMAP reconstruction results"""
    import numpy as np
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
    
    ply_path = output_dir / "colmap_point_cloud.ply"
    o3d.io.write_point_cloud(str(ply_path), pcd)
    print(f"✓ COLMAP point cloud saved: {ply_path} ({len(points)} points)")
    
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
    
    print(f"✓ Camera poses saved: {poses_path}")


def train_nerf(
    image_dir: Path,
    colmap_dir: Path,
    output_dir: Path,
    max_iterations: int,
    enable_rich: bool = False,
):
    """Train NeRF using nerfstudio (robust + log-to-file so Colab always shows progress)."""
    import os
    import shutil
    from PIL import Image, ImageFile

    ImageFile.LOAD_TRUNCATED_IMAGES = True  # ✅ tolerate truncated jpegs

    def _ensure_downscaled_images(images_dir: Path, factor: int = 4, overwrite: bool = False):
        ds_dir = images_dir.parent / f"images_{factor}"
        ds_dir.mkdir(parents=True, exist_ok=True)

        src_images = sorted([p for p in images_dir.iterdir() if _is_image_file(p)])
        dst_images = sorted([p for p in ds_dir.iterdir() if _is_image_file(p)])

        if not overwrite and len(src_images) > 0 and len(dst_images) == len(src_images):
            print(f"✓ Downscaled images already exist: {ds_dir}")
            return

        if overwrite:
            for p in ds_dir.iterdir():
                if p.is_file():
                    p.unlink()

        ok = 0
        for src in tqdm(src_images, desc=f"Downscaling x{factor}"):
            try:
                with Image.open(src) as img:
                    img = img.convert("RGB")
                    w, h = img.size
                    nw, nh = max(1, w // factor), max(1, h // factor)
                    img = img.resize((nw, nh), resample=Image.BILINEAR)
                    img.save(ds_dir / (src.stem + ".jpg"), quality=95)
                ok += 1
            except Exception as e:
                print(f"[WARN] skip bad image: {src.name} ({e})")
        print(f"✓ Created downscaled images: {ds_dir} ({ok}/{len(src_images)} ok)")

    nerf_output = output_dir / "nerf"
    data_dir = output_dir / "nerfstudio_data"
    data_dir.mkdir(parents=True, exist_ok=True)

    print("\n[1/2] Preparing data for NeRF...")

    images_dest = data_dir / "images"
    if images_dest.exists():
        shutil.rmtree(images_dest)
    shutil.copytree(image_dir, images_dest)

    downscale_factor = 4
    _ensure_downscaled_images(images_dest, factor=downscale_factor, overwrite=False)

    colmap_dest = data_dir / "colmap" / "sparse" / "0"
    colmap_dest.parent.parent.mkdir(parents=True, exist_ok=True)
    if colmap_dest.exists():
        shutil.rmtree(colmap_dest)

    if list(colmap_dir.glob("*.bin")):
        actual_colmap_dir = colmap_dir
    elif (colmap_dir / "0").exists() and list((colmap_dir / "0").glob("*.bin")):
        actual_colmap_dir = colmap_dir / "0"
    elif (colmap_dir / "models" / "0").exists() and list((colmap_dir / "models" / "0").glob("*.bin")):
        actual_colmap_dir = colmap_dir / "models" / "0"
    else:
        raise FileNotFoundError(f"Cannot find COLMAP bin files in {colmap_dir}")

    colmap_dest.mkdir(parents=True, exist_ok=True)
    for name in ["cameras.bin", "images.bin", "points3D.bin"]:
        src = actual_colmap_dir / name
        if not src.exists():
            raise FileNotFoundError(f"Missing required COLMAP file: {src}")
        shutil.copy2(src, colmap_dest / name)

    print("✓ Data prepared")

    print(f"\n[2/2] Training NeRF ({max_iterations} iterations)...")
    cmd = [
        "ns-train", "nerfacto",
        "--max-num-iterations", str(max_iterations),
        "--output-dir", str(nerf_output),
        "colmap",
        "--data", str(data_dir),
        "--auto-scale-poses", "True",
        "--downscale-factor", str(downscale_factor),
    ]

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if not enable_rich:
        env["RICH_DISABLE"] = "1"

    log_path = output_dir / "ns_train.log"
    print(f"Running: {' '.join(cmd)}")
    print(f"Logging to: {log_path}")

    try:
        with open(log_path, "w") as f:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                text=True,
                bufsize=1,
            )
            # ✅ 一邊印到 Colab output，一邊寫檔
            for line in proc.stdout:
                print(line, end="")
                f.write(line)
                f.flush()
                sys.stdout.flush()
            ret = proc.wait()
            if ret != 0:
                raise subprocess.CalledProcessError(ret, cmd)

        print(f"\n✓ NeRF training complete: {nerf_output}")
        return nerf_output

    except subprocess.CalledProcessError as e:
        print(f"\n✗ NeRF training failed: {e}")
        print(f"Check log: {log_path}")
        return None




def export_nerf_outputs(nerf_output: Path, output_dir: Path):
    print("\n" + "="*70)
    print("Exporting NeRF Dense Reconstruction")
    print("="*70)

    configs = list(nerf_output.rglob("config.yml"))
    if not configs:
        print("Warning: No config.yml found under nerf output. NeRF might not have finished.")
        return

    config_path = max(configs, key=lambda p: p.stat().st_mtime)
    print(f"Using config: {config_path}")

    # Export dense point cloud
    print("\n[1/2] Exporting dense point cloud (1M points)...")
    cmd = [
        "ns-export", "pointcloud",
        "--load-config", str(config_path),
        "--output-dir", str(output_dir),
        "--num-points", "1000000",
        "--remove-outliers", "True",
        "--use-bounding-box", "True",
    ]
    subprocess.run(cmd, check=True, stdin=subprocess.DEVNULL)
    print(f"✓ Dense point cloud exported: {output_dir / 'point_cloud.ply'}")

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
    subprocess.run(cmd, check=True, stdin=subprocess.DEVNULL)
    print(f"✓ Depth maps exported: {depth_output}")



def main():
    parser = argparse.ArgumentParser(
        description="LightGlue + hloc + NeRF Pipeline for Dense 3D Reconstruction"
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
        help="Output directory for reconstruction"
    )
    parser.add_argument(
        "--feature_type",
        type=str,
        default="disk",
        choices=["disk", "aliked-n16", "superpoint_max"],
        help="Feature extractor type"
    )
    parser.add_argument(
        "--matcher_type",
        type=str,
        default="disk+lightglue",
        choices=["disk+lightglue", "aliked+lightglue", "superpoint+lightglue"],
        help="Feature matcher type (must match feature_type)"
    )
    parser.add_argument(
        "--num_matched",
        type=int,
        default=None,
        help="Limit pairs per image (sequential). If set, overrides exhaustive pairing",
    )
    parser.add_argument(
        "--skip_nerf",
        action="store_true",
        help="Skip NeRF training, only do SfM"
    )
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=30000,
        help="Max NeRF training iterations (default: 30000)"
    )
    parser.add_argument(
        "--enable_rich",
        action="store_true",
        help="Enable rich progress bars for nerfstudio (helpful in Colab)",
    )
    
    args = parser.parse_args()
    
    # Validate feature/matcher combination
    if "disk" in args.feature_type and "disk" not in args.matcher_type:
        print("Warning: Feature type and matcher type should match!")
    
    # Run reconstruction
    model = lightglue_reconstruction_pipeline(
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        feature_type=args.feature_type,
        matcher_type=args.matcher_type,
        num_matched=args.num_matched,
        skip_nerf=args.skip_nerf,
        max_iterations=args.max_iterations,
        enable_rich=args.enable_rich,
    )
    
    # Print final summary
    print("\n" + "="*70)
    print("✓ Pipeline Completed Successfully!")
    print("="*70)
    print(f"\nResults saved in: {args.output_dir}")
    print(f"  - COLMAP point cloud: {args.output_dir / 'colmap_point_cloud.ply'}")
    print(f"  - Camera poses: {args.output_dir / 'camera_poses.txt'}")
    if not args.skip_nerf:
        print(f"  - NeRF dense point cloud: {args.output_dir / 'point_cloud.ply'} (1M points)")
        print(f"  - NeRF depth maps: {args.output_dir / 'depth_maps'}")
    print("="*70)


if __name__ == "__main__":
    main()
