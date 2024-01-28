
def detect_moving_objects(source_file, target_file, comb_dir, unique_file):
    # Load and register point clouds
    source = o3d.io.read_point_cloud(source_file)
    target = o3d.io.read_point_cloud(target_file)

    # Compute the difference between the source and target point clouds

    pcd_diff = compute_difference_point_cloud(source=source, target=target)

    # Assign a unique color to moving objects (red)
    moving_color = [1, 0, 0]  # RGB values range from 0 to 1

    # Assign a different color to static objects (grey)
    static_color = [0.5, 0.5, 0.5]  # RGB values range from 0 to 1

    # Create a new point cloud for moving objects
    moving_pcd = o3d.geometry.PointCloud()
    moving_pcd.points = pcd_diff.points
    moving_pcd.colors = o3d.utility.Vector3dVector(np.full((len(moving_pcd.points), 3), moving_color))

    # Create a new point cloud for static objects
    static_pcd = compute_difference_point_cloud(source=source, target=moving_pcd)
    static_pcd.colors = o3d.utility.Vector3dVector(np.full((len(static_pcd.points), 3), static_color))

    # Combine the point clouds
    combined_pcd = moving_pcd + static_pcd

    # Save the labeled point cloud
    o3d.io.write_point_cloud(os.path.join(align_dir, "labeled.ply"), combined_pcd)

    return combined_pcd

