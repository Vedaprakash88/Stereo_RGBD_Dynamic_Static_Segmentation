import open3d as o3d
import numpy as np
import os
from tqdm import tqdm

def register_point_clouds(source_file, target_file, align_dir, unique_file):
    # Load point clouds
    source = o3d.io.read_point_cloud(source_file)
    target = o3d.io.read_point_cloud(target_file)

    # Estimate normals
    source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=10))
    target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=10))

    # Apply point-to-plane ICP (iterative closest point)
    threshold = 0.02
    trans_init = np.asarray([[1., 0., 0., 0],
                             [0., 1., 0., 0.],
                             [0., 0., 1., 0.],
                             [0., 0., 0., 1.]])
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())

    # Transform source point cloud
    source_reg = source.transform(reg_p2p.transformation)

    # o3d.visualization.draw_geometries([source])

    # Save aligned point cloud
    o3d.io.write_point_cloud(filename=os.path.join(align_dir, unique_file + '.ply'), pointcloud=source_reg)
    return None


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
    align_save_path = os.path.join(align_dir, subfolder)
    comb_save_folder = os.path.join(labeled_dir, subfolder)
    os.makedirs(align_save_path, exist_ok=True)
    os.makedirs(comb_save_folder, exist_ok=True)
    sub_folder_len = len(os.listdir(align_save_path))

    if sub_folder_len < len(file_names_unique):
        for unique_file in tqdm(iterable=file_names_unique, desc=subfolder + ' image registration'):
            source_file = os.path.join(files_path, unique_file + '_' + f1 + '.ply')
            target_file = os.path.join(files_path, unique_file + '_' + f2 + '.ply')
            register_point_clouds(source_file=source_file, target_file=target_file, align_dir=align_save_path,
                                  unique_file=unique_file)
    if subfolder == 'training':
        for unique_file in tqdm(iterable=file_names_unique, desc='Motion Detection Progress'):
                source_file = os.path.join(align_save_path, unique_file + '.ply')
                target_file = os.path.join(files_path, unique_file + '_' + f2 + '.ply')
                comb_save_path = os.path.join(comb_save_folder, unique_file)
                # detect_motion_voxel_grid_difference(pcd1_file=source_file, pcd2_file=target_file, save_path=comb_save_path)