
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 15:29:38 2026

@author: bsbehen
"""

# This script is a complete Python pipeline for creating 3D models using an
# Intel RealSense D435 mounted on a Universal Robots UR5e. The script captures
# point cloud data, transforms captured data into a common coordinate frame,
# statistically denoises the merged point cloud using MCMD_Z, fills the base
# of the cleaned, merged cloud, and performs Poisson Surface Reconstruction to
# output a watertight STL model with a flat base.

import os
import re
import time
import copy
import cv2
import numpy as np
import open3d as o3d
import pyrealsense2 as rs
import trimesh

from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface


# =========================================================
# USER SETTINGS
# =========================================================
ROBOT_IP = "192.168.99.228"
OUT_DIR = r"C:\Users\bsbehen\OneDrive - Cal Poly Pomona\Desktop\pipeline_output"
os.makedirs(OUT_DIR, exist_ok=True)

MOVE_SPEED = 0.5
MOVE_ACCEL = 0.2
SETTLE_TIME = 1
SAVE_IMAGES = True

MIN_BOUND = [-0.2, -0.56, -0.01]
MAX_BOUND = [0.15, -0.240, 0.4]

CAPTURE_VOXEL_SIZE = 0.001

MCMD_K = 30
MCMD_H = 15
MCMD_OUTLIER_RATE = 0.5
MCMD_PR = 0.9999
MCMD_RZ_THRESHOLD = 2.5

MCMD_SAMPLE_FRACTION = 1 / 50
MCMD_RANDOM_SEED = 42

BASE_PERCENTILE = 5
BASE_NUM_BINS = 100
BASE_POINTS_PER_ROW = 100

POISSON_NORMAL_RADIUS = 0.003
POISSON_NORMAL_MAX_NN = 30
POISSON_ORIENT_K = 11
POISSON_DEPTH = 5

FLAT_BASE_PERCENTILE = 2
FLAT_BASE_BAND = 0.001
FLAT_BASE_OFFSET = 0.0005


# =========================================================
# OBJECT NAME / FILE NAMING
# =========================================================
def sanitize_name(name):
    name = name.strip().lower()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^a-z0-9_]", "", name)
    return name

OBJECT_NAME = sanitize_name(input("Enter object name: "))

MERGED_PLY = f"merged_pointcloud_{OBJECT_NAME}.ply"
MCMD_PLY = f"MCMD_cleaned_pointcloud_{OBJECT_NAME}.ply"
FILLED_PLY = f"MCMD_cleaned_pointcloud_filled_{OBJECT_NAME}.ply"
POISSON_STL = f"poisson_mesh_{OBJECT_NAME}.stl"
FLAT_STL = f"flat_base_{OBJECT_NAME}.stl"


# =========================================================
# TRANSFORM DEFINITIONS
# =========================================================
def Rodrigues(p, d):
    p = np.array(p).reshape(3, 1)
    d = np.array(d).reshape(3)

    rotation_angle = np.linalg.norm(d)
    d = d / rotation_angle

    e1 = np.array([1, 0, 0])
    e2 = np.array([0, 1, 0])
    e3 = np.array([0, 0, 1])

    def rodrigues_vector(v):
        return (
            v
            + np.sin(rotation_angle) * np.cross(d, v)
            + (1 - np.cos(rotation_angle)) * np.cross(d, np.cross(d, v))
        )

    R = np.column_stack([
        rodrigues_vector(e1),
        rodrigues_vector(e2),
        rodrigues_vector(e3)
    ])

    pose = np.eye(4)
    pose[:3, :3] = R
    pose[:3, 3:] = p
    return pose

p0, d0 = [-27.72, -380.36, 490.42], [2.353, 0.000, 0.000]
p1, d1 = [3.94, -667.25, 412.56], [2.939, 0.066, -0.070]
p2, d2 = [212.92, -494.71, 412.56], [2.515, 1.528, 0.088]
p3, d3 = [167.98, -227.46, 412.56], [1.437, 2.624, 0.226]
p4, d4 = [-85.94, -132.75, 412.56], [-0.069, 3.078, 0.311]
p5, d5 = [-294.92, -305.29, 412.56], [1.593, -2.621, -0.304]
p6, d6 = [-249.98, -572.54, 412.56], [2.617, -1.434, -0.212]

pose_tool_flange = [
    Rodrigues(p0, d0),
    Rodrigues(p1, d1),
    Rodrigues(p2, d2),
    Rodrigues(p3, d3),
    Rodrigues(p4, d4),
    Rodrigues(p5, d5),
    Rodrigues(p6, d6),
]

p_cam, d_cam = [-30, -14.9, 48.83], [0.7871, 0, 0]
pose_cam_relative_to_tool = Rodrigues(p_cam, d_cam)

pose_cam_relative_to_base = [
    pose_tool_flange[i] @ pose_cam_relative_to_tool for i in range(7)
]

pose_cam_matrices = []
for pose in pose_cam_relative_to_base:
    mat = np.array(pose, dtype=float)
    mat[:3, 3] /= 1000.0
    pose_cam_matrices.append(mat)

waypoints = [
    [p0[0] / 1000.0, p0[1] / 1000.0, p0[2] / 1000.0, d0[0], d0[1], d0[2]],
    [p1[0] / 1000.0, p1[1] / 1000.0, p1[2] / 1000.0, d1[0], d1[1], d1[2]],
    [p2[0] / 1000.0, p2[1] / 1000.0, p2[2] / 1000.0, d2[0], d2[1], d2[2]],
    [p3[0] / 1000.0, p3[1] / 1000.0, p3[2] / 1000.0, d3[0], d3[1], d3[2]],
    [p4[0] / 1000.0, p4[1] / 1000.0, p4[2] / 1000.0, d4[0], d4[1], d4[2]],
    [p5[0] / 1000.0, p5[1] / 1000.0, p5[2] / 1000.0, d5[0], d5[1], d5[2]],
    [p6[0] / 1000.0, p6[1] / 1000.0, p6[2] / 1000.0, d6[0], d6[1], d6[2]],
]


# =========================================================
# REALSENSE
# =========================================================
def start_realsense():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    pipeline.start(config)

    align = rs.align(rs.stream.color)
    profile = pipeline.get_active_profile()
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

    return pipeline, align, depth_scale, intrinsics


def save_capture_images(color_image_bgr, depth_image, save_index):
    if not SAVE_IMAGES:
        return

    cv2.imwrite(os.path.join(OUT_DIR, f"color_{save_index}.png"), color_image_bgr)

    if np.max(depth_image) > 0:
        depth_vis = (depth_image.astype(np.float32) * (255.0 / np.max(depth_image))).astype(np.uint8)
    else:
        depth_vis = depth_image.astype(np.uint8)

    cv2.imwrite(os.path.join(OUT_DIR, f"depth_{save_index}.png"), depth_vis)


def build_point_cloud_from_frames(color_image_bgr, depth_image, intrinsics, depth_scale):
    color_rgb = cv2.cvtColor(color_image_bgr, cv2.COLOR_BGR2RGB)

    color_o3d = o3d.geometry.Image(color_rgb)
    depth_o3d = o3d.geometry.Image(depth_image)

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d,
        depth_o3d,
        depth_scale=1.0 / depth_scale,
        depth_trunc=3.0,
        convert_rgb_to_intensity=False
    )

    intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(
        width=intrinsics.width,
        height=intrinsics.height,
        fx=intrinsics.fx,
        fy=intrinsics.fy,
        cx=intrinsics.ppx,
        cy=intrinsics.ppy
    )

    return o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic_o3d)


def transform_crop_downsample_save(raw_pcd, save_index, transform_matrix):
    pcd = copy.deepcopy(raw_pcd)
    pcd.transform(transform_matrix)

    aabb = o3d.geometry.AxisAlignedBoundingBox(MIN_BOUND, MAX_BOUND)
    pcd = pcd.crop(aabb)

    if CAPTURE_VOXEL_SIZE > 0 and len(pcd.points) > 0:
        pcd = pcd.voxel_down_sample(voxel_size=CAPTURE_VOXEL_SIZE)

    if len(pcd.points) > 0:
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30)
        )

    out_path = os.path.join(OUT_DIR, f"cloud_pos_{save_index}.ply")
    o3d.io.write_point_cloud(out_path, pcd)
    print(f"Saved transformed cloud {save_index}: {out_path}")

    return pcd


def capture_at_pose(pipeline, align, intrinsics, depth_scale, save_index, transform_matrix):
    for _ in range(5):
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)

    frames = pipeline.wait_for_frames()
    frames = align.process(frames)

    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()
    if not color_frame or not depth_frame:
        raise RuntimeError(f"Failed to get frames for capture {save_index}")

    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())

    save_capture_images(color_image, depth_image, save_index)

    raw_pcd = build_point_cloud_from_frames(color_image, depth_image, intrinsics, depth_scale)
    return transform_crop_downsample_save(raw_pcd, save_index, transform_matrix)


# =========================================================
# ROBOT MOTION + CAPTURE
# =========================================================
def automated_capture_loop():
    rtde_c = RTDEControlInterface(ROBOT_IP)
    rtde_r = RTDEReceiveInterface(ROBOT_IP)
    pipeline, align, depth_scale, intrinsics = start_realsense()

    pcd_list = []

    try:
        print("Connected RTDE:", rtde_r.isConnected())
        print("Current TCP pose:", rtde_r.getActualTCPPose())

        print("\nMoving to p0 (capture)...")
        rtde_c.moveL(waypoints[0], MOVE_SPEED, MOVE_ACCEL)
        time.sleep(SETTLE_TIME)
        pcd_list.append(capture_at_pose(
            pipeline, align, intrinsics, depth_scale, 0, pose_cam_matrices[0]
        ))

        print("\nMoving to p1 (capture)...")
        rtde_c.moveL(waypoints[1], MOVE_SPEED, MOVE_ACCEL)
        time.sleep(SETTLE_TIME)
        pcd_list.append(capture_at_pose(
            pipeline, align, intrinsics, depth_scale, 1, pose_cam_matrices[1]
        ))

        for i in range(2, 6):
            print("\nReturning to p0 (no capture)...")
            rtde_c.moveL(waypoints[0], MOVE_SPEED, MOVE_ACCEL)
            time.sleep(SETTLE_TIME)

            print(f"\nMoving to p{i} (capture)...")
            rtde_c.moveL(waypoints[i], MOVE_SPEED, MOVE_ACCEL)
            time.sleep(SETTLE_TIME)

            pcd_list.append(capture_at_pose(
                pipeline, align, intrinsics, depth_scale, i, pose_cam_matrices[i]
            ))

        print("\nMoving to p6 (capture)...")
        rtde_c.moveL(waypoints[6], MOVE_SPEED, MOVE_ACCEL)
        time.sleep(SETTLE_TIME)
        pcd_list.append(capture_at_pose(
            pipeline, align, intrinsics, depth_scale, 6, pose_cam_matrices[6]
        ))

        print("\nStarting reverse return path (no scans)...")

        print("\nMoving to p5 (no capture)...")
        rtde_c.moveL(waypoints[5], MOVE_SPEED, MOVE_ACCEL)
        time.sleep(SETTLE_TIME)

        print("\nReturning to p0 (no capture)...")
        rtde_c.moveL(waypoints[0], MOVE_SPEED, MOVE_ACCEL)
        time.sleep(SETTLE_TIME)

    finally:
        try:
            pipeline.stop()
        except Exception:
            pass
        try:
            rtde_c.disconnect()
        except Exception:
            pass
        try:
            rtde_r.disconnect()
        except Exception:
            pass

    merged = o3d.geometry.PointCloud()
    for pcd in pcd_list:
        merged += pcd

    merged_path = os.path.join(OUT_DIR, MERGED_PLY)
    o3d.io.write_point_cloud(merged_path, merged)
    print(f"\nSaved merged point cloud: {merged_path}")

    return merged


# =========================================================
# MCMD_Z
# =========================================================
def check_points_can_define_plane(points):
    if len(points) < 3:
        return False
    
    ref_point = points[0]
    vectors = points[1:] - ref_point
    
    v1 = None
    for v in vectors:
        if np.linalg.norm(v) > 1e-6:
            v1 = v
            break
    
    if v1 is None:
        return False
    
    for v in vectors:
        cross_mag = np.linalg.norm(np.cross(v1, v))
        if cross_mag > 1e-6:
            return True
    
    return False


def find_mcs_for_point(neighbor_points, h, I_t):
    
    # Store results from each iteration
    S_lambda0 = []      # list of lambda0 values
    S_h_indices = []    # list of corresponding h-subsets
    
    for iteration in range(I_t):
        # -------------------------
        # Step 1: Find valid h0 subset (non-collinear points)
        # -------------------------
        h0 = 3
        max_attempts = 100
        valid_subset_found = False
        attempt = 0
        
        while not valid_subset_found and attempt < max_attempts:
            attempt += 1
            
            # Randomly select h0 points
            subset_indices = np.random.choice(len(neighbor_points), h0, replace=False)
            subset = neighbor_points[subset_indices]
            
            # Check if these points can define a plane
            can_define_plane = check_points_can_define_plane(subset)
            
            if can_define_plane:
                valid_subset_found = True
            else:
                h0 += 1  # Add another point and try again
        
        if not valid_subset_found:
            continue
        
        # -------------------------
        # Step 2: Fit plane to subset, calculate ODs for ALL points
        # -------------------------
        # Fit plane to subset
        centered_subset = subset - np.mean(subset, axis=0)
        cov = (centered_subset.T @ centered_subset) / len(subset)
        U, S, Vt = np.linalg.svd(cov)
        normal = U[:, -1]  # smallest eigenvector
        mean_point = np.mean(subset, axis=0)
        
        # Calculate ODs for all neighborhood points
        ODs = []
        for point in neighbor_points:
            v = point - mean_point
            od = abs(np.dot(v, normal))
            ODs.append(od)
        ODs = np.array(ODs)
        
        # Sort by OD
        sorted_idx = np.argsort(ODs)
        
        # -------------------------
        # Step 3: Take h smallest, fit new plane, get lambda0
        # -------------------------
        h_indices = sorted_idx[:h]
        h_points = neighbor_points[h_indices]
        
        # Fit plane to h_points
        centered_h = h_points - np.mean(h_points, axis=0)
        cov_h = (centered_h.T @ centered_h) / len(centered_h)
        U_h, S_h, Vt_h = np.linalg.svd(cov_h)
        lambda0 = S_h[-1]  # smallest eigenvalue
        
        # Store results
        S_lambda0.append(lambda0)
        S_h_indices.append(h_indices)
    
    # -------------------------
    # Find best MCS across iterations
    # -------------------------
    if len(S_lambda0) > 0:
        best_idx = np.argmin(S_lambda0)
        best_h_subset = S_h_indices[best_idx]
        return best_h_subset
    else:
        return None


def MCMD_z(neighbor_points, mcs_global_indices, neighbor_global_indices):
    # Step 1: Get the MCS points (trusted set)
    # Find which positions in neighbor_points correspond to MCS
    mcs_mask = np.isin(neighbor_global_indices, mcs_global_indices)
    mcs_points = neighbor_points[mcs_mask]
    
    # Step 2: Fit plane to MCS
    robust_mean = np.mean(mcs_points, axis=0)
    centered_mcs = mcs_points - robust_mean
    cov = (centered_mcs.T @ centered_mcs) / len(mcs_points)
    U, S, Vt = np.linalg.svd(cov)
    normal = U[:, -1]  # smallest eigenvector
    
    # Step 3: Calculate orthogonal distances for ALL points
    ODs = []
    for point in neighbor_points:
        v = point - robust_mean
        od = abs(np.dot(v, normal))
        ODs.append(od)
    ODs = np.array(ODs)
    
    # Step 4: Robust z-scores on plane distances (median + MAD)
    med = np.median(ODs)
    mad = 1.4826 * np.median(np.abs(ODs - med))
    rz = np.abs(ODs - med) / mad
    
    # Step 5: Classify based on threshold
    inlier_mask = rz < MCMD_RZ_THRESHOLD
    outlier_mask = ~inlier_mask
    
    # Convert to global indices
    inlier_indices = neighbor_global_indices[inlier_mask]
    outlier_indices = neighbor_global_indices[outlier_mask]
    
    return inlier_indices, outlier_indices


def run_mcmd_z(pcd):
    # -------------------------
    # Load point cloud
    # -------------------------
    print("Loading point cloud...")
    print(f"Loaded cloud with {len(pcd.points)} points")
    
    # Convert to numpy array for easier manipulation
    points = np.asarray(pcd.points, dtype=np.float64)
    print(f"Points shape: {points.shape}")  # Should be (N, 3)
    
    # Keep colors if they exist
    has_colors = len(pcd.colors) == len(points)
    if has_colors:
        colors = np.asarray(pcd.colors)
    else:
        colors = None
    
    # -------------------------
    # Step 0: Build k-d tree for finding neighbors
    # -------------------------
    print("\n--- Step 0: Building k-d tree ---")
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    
    # -------------------------
    # Parameters
    # -------------------------
    k = MCMD_K   # neighborhood size
    h = MCMD_H   # size of candidate MCS
    
    outlier_rate = MCMD_OUTLIER_RATE
    Pr = MCMD_PR
    I_t = int(np.round(np.log(1 - Pr) / np.log(1 - (1 - outlier_rate)**3)))  # iterations per point
    
    sample_fraction = MCMD_SAMPLE_FRACTION
    random_seed = MCMD_RANDOM_SEED
    
    print(f"\nParameters: k={k}, h={h}, I_t={I_t} iterations per point")
    
    # -------------------------
    # Main loop: Process a SAMPLED subset of points as MCS centers
    # -------------------------
    print("\n" + "="*60)
    print("Finding MCS for sampled center points")
    
    # Sample center points
    rng = np.random.default_rng(random_seed)
    num_points = len(points)
    num_sampled = max(1, int(np.ceil(sample_fraction * num_points)))
    sampled_point_indices = np.sort(rng.choice(num_points, size=num_sampled, replace=False))
    
    print(f"Sampling {num_sampled}/{num_points} points as centers")
    
    # Store MCS for each sampled point (as indices into the original point cloud)
    all_mcs_indices = []  # each entry will be an array of indices
    all_inliers = []
    all_outliers = [] 
    
    for loop_idx, point_idx in enumerate(sampled_point_indices):
        print(f"\n--- Processing sampled point {loop_idx + 1}/{num_sampled} (index {point_idx}) ---")
        
        # Get current point
        current_point = points[point_idx]
        
        # Find its k nearest neighbors
        [k_found, neighbor_indices, _] = pcd_tree.search_knn_vector_3d(current_point, k)
        neighbor_points = points[np.asarray(neighbor_indices)]
        
        # Find MCS for this point's neighborhood
        mcs_local_indices = find_mcs_for_point(neighbor_points, h, I_t)
        
        # STORE THE RESULTS!
        if mcs_local_indices is not None:
            neighbor_indices_array = np.asarray(neighbor_indices)
            mcs_global_indices = neighbor_indices_array[mcs_local_indices]
            all_mcs_indices.append(mcs_global_indices)
            
            inliers, outliers = MCMD_z(neighbor_points, mcs_global_indices, neighbor_indices_array)
        
            print(f"  Inliers: {len(inliers)}, Outliers: {len(outliers)}")
        
            # Store results
            all_inliers.append(inliers)
            all_outliers.append(outliers)
            
        else:
            all_mcs_indices.append(None)
            all_inliers.append(None)
            all_outliers.append(None)
    
    
    # Count votes
    vote_inlier = np.zeros(len(points))
    vote_outlier = np.zeros(len(points))
    
    for inliers, outliers in zip(all_inliers, all_outliers):
        if inliers is not None:
            vote_inlier[inliers] += 1
            vote_outlier[outliers] += 1
    
    # Remove outliers
    final_inlier_mask = vote_inlier > 2*vote_outlier
    
    # Optional fallback: if a point got no votes, keep it
    zero_vote_mask = (vote_inlier == 0) & (vote_outlier == 0)
    final_inlier_mask[zero_vote_mask] = True
    
    # Create cleaned point cloud
    cleaned_points = points[final_inlier_mask]
    cleaned_pcd = o3d.geometry.PointCloud()
    cleaned_pcd.points = o3d.utility.Vector3dVector(cleaned_points)
    
    # Preserve colors if the input had them
    if colors is not None:
        cleaned_colors = colors[final_inlier_mask]
        cleaned_pcd.colors = o3d.utility.Vector3dVector(cleaned_colors)
    
    # Save it
    cleaned_path = os.path.join(OUT_DIR, MCMD_PLY)
    o3d.io.write_point_cloud(cleaned_path, cleaned_pcd)
    
    # Number of points (not total elements)
    n_original = len(points)
    n_cleaned = len(cleaned_points)
    print(f"Original: {n_original} points")
    print(f"Cleaned: {n_cleaned} points")
    print(f"Removed: {n_original - n_cleaned} points")
    print("Saved:", cleaned_path)
    
    return cleaned_pcd


# =========================================================
# BASE FILLING
# =========================================================
def fill_base(pcd):
    pcd = copy.deepcopy(pcd)

    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors, dtype=np.float64) if pcd.has_colors() else None

    z_min = np.min(points[:, 2])
    z_threshold = np.percentile(points[:, 2], BASE_PERCENTILE)
    bottom_points = points[points[:, 2] <= z_threshold]

    xy_bottom = bottom_points[:, :2]
    x_values = xy_bottom[:, 0]

    x_bins = np.linspace(np.min(x_values), np.max(x_values), BASE_NUM_BINS + 1)
    bin_indices = np.digitize(x_values, x_bins) - 1

    base_points = []
    for bin_idx in range(BASE_NUM_BINS):
        mask = (bin_indices == bin_idx)
        if np.sum(mask) > 0:
            bin_points = xy_bottom[mask]
            x_avg = np.mean(bin_points[:, 0])
            y_min = np.min(bin_points[:, 1])
            y_max = np.max(bin_points[:, 1])

            y_fill = np.linspace(y_min, y_max, BASE_POINTS_PER_ROW)
            for y in y_fill:
                base_points.append([x_avg, y, z_min])

    base_points = np.array(base_points)

    base_pcd = o3d.geometry.PointCloud()
    base_pcd.points = o3d.utility.Vector3dVector(base_points)

    if len(base_points) > 0:
        base_colors = np.tile(np.array([[1.0, 0.0, 0.0]]), (len(base_points), 1))
        base_pcd.colors = o3d.utility.Vector3dVector(base_colors)

    if colors is not None and len(colors) == len(points):
        pcd.colors = o3d.utility.Vector3dVector(colors)

    combined = pcd + base_pcd

    filled_path = os.path.join(OUT_DIR, FILLED_PLY)
    o3d.io.write_point_cloud(filled_path, combined)
    print("Saved:", filled_path)

    return combined


# =========================================================
# POISSON
# =========================================================
def run_poisson(pcd):
    pcd_for_mesh = copy.deepcopy(pcd)
    pcd_for_mesh.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=POISSON_NORMAL_RADIUS,
            max_nn=POISSON_NORMAL_MAX_NN
        )
    )
    pcd_for_mesh.orient_normals_consistent_tangent_plane(POISSON_ORIENT_K)
    pcd_for_mesh.normalize_normals()

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd_for_mesh, depth=POISSON_DEPTH
    )
    mesh.compute_vertex_normals()

    mesh_path = os.path.join(OUT_DIR, POISSON_STL)
    o3d.io.write_triangle_mesh(mesh_path, mesh)
    print("Saved:", mesh_path)

    return mesh_path


# =========================================================
# FLAT BASE
# =========================================================
def flatten_mesh_base(mesh_path):
    mesh = trimesh.load(mesh_path)
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump(concatenate=True)

    z_vals = mesh.vertices[:, 2]
    z_base = np.percentile(z_vals, FLAT_BASE_PERCENTILE)
    band = z_vals[z_vals <= z_base + FLAT_BASE_BAND]
    z_ref = band.mean()
    cut_z = z_ref + FLAT_BASE_OFFSET

    cut_mesh = mesh.slice_plane(
        plane_origin=np.array([0.0, 0.0, cut_z]),
        plane_normal=np.array([0.0, 0.0, 1.0]),
        cap=True,
        engine="earcut"
    )

    cut_mesh.merge_vertices()
    cut_mesh.remove_unreferenced_vertices()
    cut_mesh.fix_normals()

    final_mesh_path = os.path.join(OUT_DIR, FLAT_STL)
    cut_mesh.export(final_mesh_path)
    print("Saved:", final_mesh_path)

    return final_mesh_path


# =========================================================
# MAIN
# =========================================================
def main():
    print(f"\nObject name: {OBJECT_NAME}")
    print("Files that will be saved:")
    print(" ", MERGED_PLY)
    print(" ", MCMD_PLY)
    print(" ", FILLED_PLY)
    print(" ", POISSON_STL)
    print(" ", FLAT_STL)

    merged_pcd = automated_capture_loop()
    cleaned_pcd = run_mcmd_z(merged_pcd)
    filled_pcd = fill_base(cleaned_pcd)
    poisson_mesh_path = run_poisson(filled_pcd)
    final_mesh_path = flatten_mesh_base(poisson_mesh_path)

    print("\nPIPELINE COMPLETE")
    print("Final STL:", final_mesh_path)


if __name__ == "__main__":
    main()
