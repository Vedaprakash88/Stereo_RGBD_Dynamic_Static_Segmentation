import open3d as o3d
import numpy as np
import os
from tqdm import tqdm

def register_point_clouds(source_file, target_file, align_dir, unique_file):
    # Load point clouds
    source = o3d.io.read_point_cloud(source_file)
    target = o3d.io.read_point_cloud(target_file)

    # Downsample the point clouds (this down sampling is creating just one point cloud,  it’s likely that the voxel grid
    #     # size for downsampling is too large. Therefore, ignored this step)
    # source = source.voxel_down_sample(voxel_size=0.001)
    # target = target.voxel_down_sample(voxel_size=0.001)


    # Estimate normals
    source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=10))
    target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=10))

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

    # o3d.visualization.draw_geometries([source])

    # Save aligned point cloud
    o3d.io.write_point_cloud(filename=os.path.join(align_dir, unique_file + '.ply'), pointcloud=source)

    return None

def detect_moving_objects(source_file, target_file):
    # Load and register point clouds
    source = register_point_clouds(source_file, target_file)

    # Load the target point cloud
    target = o3d.io.read_point_cloud(target_file)

    # Compute the difference between the source and target point clouds
    pcd_diff = source - target

    # Assign a unique color to moving objects (red)
    moving_color = [1, 0, 0]  # RGB values range from 0 to 1

    # Assign a different color to static objects (grey)
    static_color = [0.5, 0.5, 0.5]  # RGB values range from 0 to 1

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
    o3d.io.write_point_cloud(os.path.join(align_dir, "labeled.ply"), combined_pcd)

    return combined_pcd

ply_dir = "D:\\10. SRH_Academia\\1. All_Notes\\4. Thesis\\5. WIP\\Data\\KITTI_Motion\\data_scene_flow\\PLY"
align_dir = "D:\\10. SRH_Academia\\1. All_Notes\\4. Thesis\\5. WIP\\Data\\KITTI_Motion\\data_scene_flow\\aligned"
labeled_dir = "D:\\10. SRH_Academia\\1. All_Notes\\4. Thesis\\5. WIP\\Data\\KITTI_Motion\\data_scene_flow\\labeled"
subfolders = ["testing", "training"]
f1 = '10'
f2 = '11'
os.makedirs(align_dir, exist_ok=True)
os.makedirs(labeled_dir, exist_ok=True)

for subfolder in subfolders:
    files_path = os.path.join(ply_dir, subfolder)
    file_names = os.listdir(files_path)
    file_names_unique = {each_file[:6] for each_file in file_names}
    save_path = os.path.join(align_dir, subfolder)
    os.makedirs(save_path, exist_ok=True)

    for unique_file in tqdm(iterable=file_names_unique, desc=subfolder + ' image registration'):
        source_file = os.path.join(files_path, unique_file + '_' + f1 + '.ply')
        target_file = os.path.join(files_path, unique_file + '_' + f2 + '.ply')
        register_point_clouds(source_file=source_file, target_file=target_file, align_dir=save_path,
                              unique_file=unique_file)
        detect_moving_objects(source_file=source_file, target_file=target_file, align_dir=save_path,
                              unique_file=unique_file)

