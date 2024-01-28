import os
import pandas as pd
import open3d as o3d
from tqdm import tqdm

class Rgb23D:
    @staticmethod
    def get_paths():
        # Directories
        rgb_dir = "D:\\10. SRH_Academia\\1. All_Notes\\4. Thesis\\5. WIP\\Data\\KITTI_Motion\\data_scene_flow\\RGB_images"
        depth_dir = "D:\\10. SRH_Academia\\1. All_Notes\\4. Thesis\\5. WIP\\Data\\KITTI_Motion\\data_scene_flow\\Depth_images"
        ply_dir = "D:\\10. SRH_Academia\\1. All_Notes\\4. Thesis\\5. WIP\\Data\\KITTI_Motion\\data_scene_flow\\PLY"
        error_folder = "D:\\10. SRH_Academia\\1. All_Notes\\4. Thesis\\5. WIP\\Data\\KITTI_Motion\\data_scene_flow\\Error_Handling"
        error_df = pd.DataFrame()
        sub_dirs = ["testing", "training"]
        os.makedirs(ply_dir, exist_ok=True)
        return rgb_dir, depth_dir, ply_dir, sub_dirs, error_folder, error_df

    @staticmethod
    def visualize_ply(file_path):
        # Load the .ply file
        ply = o3d.io.read_point_cloud(file_path)

        # Visualize the point cloud
        o3d.visualization.draw_geometries([ply])

    @staticmethod
    def read_images(rgb_dir, depth_dir, sub_dir, file):
        # Get list of files in each directory
        rgb_image = o3d.io.read_image(os.path.join(rgb_dir, sub_dir, file))

        # Load depth image
        depth_image = o3d.io.read_image(os.path.join(depth_dir, sub_dir, file))

        return rgb_image, depth_image

    @staticmethod
    def create_point_cloud(rgb_image, depth_image, ply_dir, sub_dir, file):
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_image, depth_image,
                                                                        convert_rgb_to_intensity=False)

        # Create point cloud
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image,
                                                             o3d.camera.PinholeCameraIntrinsic(
                                                                 o3d.camera.PinholeCameraIntrinsicParameters.
                                                                 PrimeSenseDefault))

        # Flip it, otherwise the pointcloud will be upside down
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        # Save point cloud
        o3d.io.write_point_cloud(os.path.join(ply_dir, sub_dir, file.replace('.png', '.ply')), pcd)
        return f'{file} point cloud created'