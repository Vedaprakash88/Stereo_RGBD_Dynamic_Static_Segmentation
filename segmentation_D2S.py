import open3d as o3d
import numpy as np

def register_point_clouds(source_file, target_file):
    # Load point clouds
    source = o3d.io.read_point_cloud(source_file)
    target = o3d.io.read_point_cloud(target_file)

    # Downsample the point clouds
    source = source.voxel_down_sample(voxel_size=0.05)
    target = target.voxel_down_sample(voxel_size=0.05)

    # Estimate normals
    source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # Apply point-to-plane ICP
    threshold = 0.02
    trans_init = np.asarray([[1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]])
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())

    # Transform source point cloud
    source.transform(reg_p2p.transformation)

    # Save aligned point cloud
    o3d.io.write_point_cloud("aligned.ply", source)

    return source


def detect_moving_objects(source_file, target_file):
    # Load and register point clouds
    source = register_point_clouds(source_file, target_file)

    # Load the target point cloud
    target = o3d.io.read_point_cloud(target_file)

    # Compute the difference between the source and target point clouds
    pcd_diff = source - target

    # Assign a unique color to moving objects (e.g., red)
    moving_color = [1, 0, 0]  # RGB values range from 0 to 1

    # Assign a different color to static objects (e.g., green)
    static_color = [0, 1, 0]  # RGB values range from 0 to 1

    # Create a new point cloud for moving objects
    moving_pcd = o3d.geometry.PointCloud()
    moving_pcd.points = pcd_diff.points
    moving_pcd.colors = o3d.utility.Vector3dVector(np.full((len(moving_pcd.points), 3), moving_color))

    # Create a new point cloud for static objects
    static_pcd = source - moving_pcd
    static_pcd.colors = o3d.utility.Vector3dVector(np.full((len(static_pcd.points), 3), static_color))

    # Combine the point clouds
    combined_pcd = moving_pcd + static_pcd

    # Save the labeled point cloud
    o3d.io.write_point_cloud("labeled.ply", combined_pcd)

    return combined_pcd


