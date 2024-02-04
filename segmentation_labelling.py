import copy
import open3d as o3d
import numpy as np
import os
from tqdm import tqdm
from matplotlib import pyplot as plt

def get_translation():
    trans_init = np.eye(4)
    trans_init[:3, 3] = np.array([0.54, 0., 0.])
    return trans_init

def show_reg_status(source, target, trans_init, threshold):
    # We have to start with global transformation before we do ICT transformation, the code below shows
    # how the result looks like
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)

    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])

    source_temp.transform(trans_init)
    # o3d.visualization.draw_plotly([source_temp, target_temp], zoom=0.5)

    # evaluation of registration

    evaluation = o3d.pipelines.registration.evaluate_registration(
        source_temp, target_temp, threshold, trans_init
    )
    print(evaluation)

def register_point_clouds(source_file, target_file, align_dir, unique_file):
    # Load point clouds
    source = o3d.io.read_point_cloud(source_file)
    target = o3d.io.read_point_cloud(target_file)

    # Extract Translation
    trans_init = get_translation()
    threshold = 0.01
    # show_reg_status(source, target, trans_init, threshold)

    # Estimate normals
    source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # Apply point-to-plane ICP (iterative closest point)

    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())

    # Transform source point cloud
    source = source.transform(reg_p2p.transformation)

    # o3d.visualization.draw_geometries([source])

    # Save aligned point cloud
    o3d.io.write_point_cloud(filename=os.path.join(align_dir, unique_file + '.pcd'), pointcloud=source)

def detect_moving_objects(registered_point_cloud, save_path, distance_threshold=5):
    # Convert point cloud to numpy array
    point_cloud_np = np.asarray(registered_point_cloud.points)

    # Calculate Euclidean distance between consecutive points
    distances = np.linalg.norm(np.diff(point_cloud_np, axis=0), axis=1)

    # Create a mask for moving points based on the distance threshold
    moving_points_mask = np.concatenate(([False], distances > distance_threshold))

    # Extract moving and static points
    moving_points = point_cloud_np[moving_points_mask]
    static_points = point_cloud_np[~moving_points_mask]

    # Extract colors
    colors = np.asarray(registered_point_cloud.colors)
    moving_colors = colors[moving_points_mask]
    static_colors = colors[~moving_points_mask]

    # Create separate point clouds for moving and static points
    moving_point_cloud = o3d.geometry.PointCloud()
    moving_point_cloud.points = o3d.utility.Vector3dVector(moving_points)
    moving_point_cloud.colors = o3d.utility.Vector3dVector(moving_colors)

    static_point_cloud = o3d.geometry.PointCloud()
    static_point_cloud.points = o3d.utility.Vector3dVector(static_points)
    static_point_cloud.colors = o3d.utility.Vector3dVector(static_colors)

    # Combine moving and static point clouds
    combined_point_cloud = o3d.geometry.PointCloud()
    combined_point_cloud.points = o3d.utility.Vector3dVector(
        np.vstack([moving_point_cloud.points, static_point_cloud.points]))
    combined_point_cloud.colors = o3d.utility.Vector3dVector(
        np.vstack([moving_point_cloud.colors, static_point_cloud.colors]))

    # Visualize the combined result
    o3d.visualization.draw_geometries([combined_point_cloud])

    # Save the combined point cloud to a file
    o3d.io.write_point_cloud(save_path, combined_point_cloud)


# Example usage

ply_dir = "D:\\10. SRH_Academia\\1. All_Notes\\4. Thesis\\5. WIP\\Data\\KITTI_Motion\\data_scene_flow\\CV2_PCD"
align_dir = "D:\\10. SRH_Academia\\1. All_Notes\\4. Thesis\\5. WIP\\Data\\KITTI_Motion\\data_scene_flow\\aligned"
labeled_dir = "D:\\10. SRH_Academia\\1. All_Notes\\4. Thesis\\5. WIP\\Data\\KITTI_Motion\\data_scene_flow\\labeled"
calib_folder = "D:\\10. SRH_Academia\\1. All_Notes\\4. Thesis\\5. WIP\\Data\\KITTI_Motion\\data_scene_flow_calib\\calib_cam_to_cam"
subfolders = ["testing", "training"]
f1 = '10'
f2 = '11'
os.makedirs(align_dir, exist_ok=True)
os.makedirs(labeled_dir, exist_ok=True)

for subfolder in subfolders:
    files_path = os.path.join(ply_dir, subfolder)
    file_names = os.listdir(files_path)
    file_names_unique = {each_file[:6] for each_file in file_names}
    align_save_path = os.path.join(align_dir, subfolder)
    comb_save_folder = os.path.join(labeled_dir, subfolder)
    os.makedirs(align_save_path, exist_ok=True)
    os.makedirs(comb_save_folder, exist_ok=True)
    sub_folder_len = len(os.listdir(align_save_path))
    calib_path = os.path.join(calib_folder, subfolder)

    if sub_folder_len < len(file_names_unique):
        for unique_file in tqdm(iterable=file_names_unique, desc=subfolder + ' image registration'):
            source_file = os.path.join(files_path, unique_file + '_' + f1 + '.pcd')
            target_file = os.path.join(files_path, unique_file + '_' + f2 + '.pcd')
            calib_file = os.path.join(calib_path, unique_file + '.txt')
            register_point_clouds(source_file=source_file, target_file=target_file, align_dir=align_save_path,
                                  unique_file=unique_file)
    if subfolder == 'training' and sub_folder_len == len(file_names_unique):
        for unique_file in tqdm(iterable=file_names_unique, desc='Motion Detection Progress'):
            source_file = os.path.join(align_save_path, unique_file + '.pcd')
            comb_save_path = os.path.join(comb_save_folder, unique_file + '.pcd')
            detect_moving_objects(registered_point_cloud=source_file, save_path=comb_save_path)