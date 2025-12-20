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


def lightglue_reconstruction_pipeline(
    image_dir: Path,
    output_dir: Path,
    feature_type: str = "disk",  # disk, aliked-n16, superpoint_max
    matcher_type: str = "disk+lightglue",  # disk+lightglue, aliked+lightglue
    num_matched: int = None,
    skip_nerf: bool = False,
    max_iterations: int = 30000,
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
        if p.suffix.lower() in ['.jpg', '.jpeg', '.png']
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
    pairs_from_exhaustive.main(pairs_path, image_list=image_list)
    
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


def train_nerf(image_dir: Path, colmap_dir: Path, output_dir: Path, max_iterations: int):
    """Train NeRF using nerfstudio"""
    nerf_output = output_dir / "nerf"
    
    # Prepare data directory
    data_dir = output_dir / "nerfstudio_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n[1/2] Preparing data for NeRF...")
    
    # Copy images
    import shutil
    images_dest = data_dir / "images"
    if images_dest.exists():
        shutil.rmtree(images_dest)
    shutil.copytree(image_dir, images_dest)
    
    # Copy COLMAP results (nerfstudio expects colmap/sparse/0)
    colmap_dest = data_dir / "colmap" / "sparse" / "0"
    colmap_dest.parent.parent.mkdir(parents=True, exist_ok=True)
    if colmap_dest.exists():
        shutil.rmtree(colmap_dest)
    
    # Detect COLMAP directory structure
    # hloc outputs to sparse/models/0/, but bin files may be in sparse/ root
    if list(colmap_dir.glob("*.bin")):
        # Bin files are directly in colmap_dir
        actual_colmap_dir = colmap_dir
    elif (colmap_dir / "0").exists() and list((colmap_dir / "0").glob("*.bin")):
        actual_colmap_dir = colmap_dir / "0"
    elif (colmap_dir / "models" / "0").exists() and list((colmap_dir / "models" / "0").glob("*.bin")):
        actual_colmap_dir = colmap_dir / "models" / "0"
    else:
        raise FileNotFoundError(f"Cannot find COLMAP bin files in {colmap_dir}")
    
    print(f"Copying COLMAP from: {actual_colmap_dir}")
    colmap_dest.mkdir(parents=True, exist_ok=True)
    
    # Copy only the necessary bin files
    for bin_file in ['cameras.bin', 'images.bin', 'points3D.bin', 'frames.bin', 'rigs.bin']:
        src = actual_colmap_dir / bin_file
        if src.exists():
            shutil.copy2(src, colmap_dest / bin_file)
    
    print("✓ Data prepared")
    
    # Train NeRF
    print(f"\n[2/2] Training NeRF ({max_iterations} iterations)...")
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
    print("📊 NeRF训练进度（实时显示）：")
    print("="*70)
    try:
        # 直接显示到终端，不捕获输出
        subprocess.run(cmd, check=True)
        print(f"\n✓ NeRF training complete: {nerf_output}")
        return nerf_output
    except subprocess.CalledProcessError as e:
        print(f"\n✗ NeRF training failed: {e}")
        return None


def export_nerf_outputs(nerf_output: Path, output_dir: Path):
    """Export NeRF dense point cloud and depth maps"""
    print("\n" + "="*70)
    print("Exporting NeRF Dense Reconstruction")
    print("="*70)
    
    # Find trained model
    model_dirs = list(nerf_output.glob("nerfacto/*/"))
    if not model_dirs:
        print("Warning: No trained NeRF model found")
        return
    
    latest_model = sorted(model_dirs)[-1]
    config_path = latest_model / "config.yml"
    
    # Export dense point cloud (1M points)
    print("\n[1/2] Exporting dense point cloud (1M points)...")
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
        print(f"✓ Dense point cloud exported: {output_dir / 'point_cloud.ply'}")
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
        print(f"✓ Depth maps exported: {depth_output}")
    except subprocess.CalledProcessError as e:
        print(f"✗ Depth map export failed: {e}")


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
        skip_nerf=args.skip_nerf,
        max_iterations=args.max_iterations,
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
