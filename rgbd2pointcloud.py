import os
import pandas as pd
import open3d as o3d
from tqdm import tqdm


# Directories

class Rgb23D():
    @staticmethod
    def get_paths() -> object:
        rgb_dir = "D:\\10. SRH_Academia\\1. All_Notes\\4. Thesis\\5. WIP\\Data\\KITTI_Motion\\RGB_images"
        depth_dir = "D:\\10. SRH_Academia\\1. All_Notes\\4. Thesis\\5. WIP\\Data\\KITTI_Motion\\Depth_images"
        ply_dir = "D:\\10. SRH_Academia\\1. All_Notes\\4. Thesis\\5. WIP\\Data\\KITTI_Motion\\PLY"
        error_folder = "D:\\10. SRH_Academia\\1. All_Notes\\4. Thesis\\5. WIP\\Data\\KITTI_Motion\\Error_Handling"
        os.makedirs(ply_dir, exist_ok=True)
        os.makedirs(error_folder, exist_ok=True)
        sub_dirs = ["training", "testing"]
        error_df = pd.DataFrame()
        return rgb_dir, depth_dir, ply_dir, sub_dirs, error_folder, error_df

    @staticmethod
    def visualize_ply(file_path):
        # Load the .ply file
        ply = o3d.io.read_point_cloud(file_path)

        # Visualize the point cloud
        o3d.visualization.draw_geometries([ply])

    @staticmethod
    def create_point_cloud(rgb_image, depth_image, ply_dir, sub_dir, file):

        # Create RGBD image
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_image, depth_image,
                                                                        convert_rgb_to_intensity=False)

        # Create point cloud
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image,
                                                             o3d.camera.PinholeCameraIntrinsic(
                                                                 o3d.camera.PinholeCameraIntrinsicParameters.
                                                                 PrimeSenseDefault))
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        o3d.io.write_point_cloud(os.path.join(ply_dir, sub_dir, file.replace('.png', '.ply')), pcd)

        return (f'{file} pcd created')

    @staticmethod
    def read_images(rgb_dir, depth_dir, sub_dir, file):
        rgb_image = o3d.io.read_image(os.path.join(rgb_dir, sub_dir, file))
        # Load depth image
        depth_image = o3d.io.read_image(os.path.join(depth_dir, sub_dir, file))
        return rgb_image, depth_image