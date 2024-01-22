import os
import numpy as np
import open3d as o3d
from tqdm import tqdm
# Directories
rgb_dir = "D:\\10. SRH_Academia\\1. All_Notes\\4. Thesis\\5. WIP\\Data\\KITTI\\archive\\RGB_images"
depth_dir = "D:\\10. SRH_Academia\\1. All_Notes\\4. Thesis\\5. WIP\\Data\\KITTI\\archive\\Depth_images"
ply_dir = "D:\\10. SRH_Academia\\1. All_Notes\\4. Thesis\\5. WIP\\Data\\KITTI\\archive\\PLY"

# Subdirectories
sub_dirs = ["training", "testing"]

def visualize_ply(file_path):
    # Load the .ply file
    ply = o3d.io.read_point_cloud(file_path)

    # Visualize the point cloud
    o3d.visualization.draw_geometries([ply])


for sub_dir in sub_dirs:
    # Get list of files in each directory
    rgb_files = os.listdir(os.path.join(rgb_dir, sub_dir))
    depth_files = os.listdir(os.path.join(depth_dir, sub_dir))

    # Find common files in both directories
    common_files = list(set(rgb_files) & set(depth_files))

    for file in tqdm(common_files):
        # Load RGB image
        rgb_image = o3d.io.read_image(os.path.join(rgb_dir, sub_dir, file))

        # Load depth image
        depth_image = o3d.io.read_image(os.path.join(depth_dir, sub_dir, file))

        # Create RGBD image
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_image, depth_image, convert_rgb_to_intensity=False)

        # Create point cloud
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image,
                                                             o3d.camera.PinholeCameraIntrinsic(
                                                                 o3d.camera.PinholeCameraIntrinsicParameters.
                                                                 PrimeSenseDefault))

        # Save point cloud
        o3d.io.write_point_cloud(os.path.join(ply_dir, sub_dir, file.replace('.png', '.ply')), pcd)

visualize_ply("/path/to/your/file.ply")