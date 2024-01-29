import os
import pandas as pd
import open3d as o3d
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

class Rgb23D:
    @staticmethod
    def get_paths():
        # Directories
        rgb_dir = "D:\\10. SRH_Academia\\1. All_Notes\\4. Thesis\\5. WIP\\Data\\KITTI_Motion\\data_scene_flow\\RGB_images"
        depth_dir = "D:\\10. SRH_Academia\\1. All_Notes\\4. Thesis\\5. WIP\\Data\\KITTI_Motion\\data_scene_flow\\Depth_images"
        ply_dir = "D:\\10. SRH_Academia\\1. All_Notes\\4. Thesis\\5. WIP\\Data\\KITTI_Motion\\data_scene_flow\\PCD"
        error_folder = "D:\\10. SRH_Academia\\1. All_Notes\\4. Thesis\\5. WIP\\Data\\KITTI_Motion\\data_scene_flow\\Error_Handling"
        calib_folder = "D:\\10. SRH_Academia\\1. All_Notes\\4. Thesis\\5. WIP\\Data\\KITTI_Motion\\data_scene_flow_calib\\calib_cam_to_cam"
        error_df = pd.DataFrame()
        sub_dirs = ["testing", "training"]
        os.makedirs(ply_dir, exist_ok=True)
        return rgb_dir, depth_dir, ply_dir, sub_dirs, error_folder, error_df, calib_folder

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
        print(depth_image)
        print(rgb_image)

        return rgb_image, depth_image

    @staticmethod
    def create_point_cloud(rgb_image, depth_image, ply_dir, sub_dir, file, calibration_params):
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_image, depth_image,
                                                                        convert_rgb_to_intensity=True)
        # plt.subplot(1, 2, 1)
        # plt.title('greyscale image')
        # plt.imshow(rgbd_image.color)
        # plt.subplot(1, 2, 2)
        # plt.title('depth image')
        # plt.imshow(rgbd_image.depth)
        # plt.show()

        # Intrinsic parameters (with reference to the 00001.txt)

        fx = calibration_params[0]  # Focal length in x-axis
        fy = calibration_params[1]  # Focal length in y-axis
        cx = calibration_params[2]  # Principal point in x-axis
        cy = calibration_params[3]  # Principal point in y-axis


        # fx = 7.215377e+02  # Focal length in x-axis
        # fy = 7.215377e+02  # Focal length in y-axis
        # cx = 6.095593e+02  # Principal point in x-axis
        # cy = 1.728540e+02  # Principal point in y-axis

        # Get the width and height of the image
        height, width, _ = rgb_image.shape

        # Create a PinholeCameraIntrinsic object using the intrinsic parameters
        intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

        # Create point cloud
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)

        # Flip it, otherwise the pointcloud will be upside down
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        o3d.visualization.draw_plotly([pcd], zoom=0.3412)

        # Save point cloud
        o3d.io.write_point_cloud(os.path.join(ply_dir, sub_dir, file.replace('.png', '.pcd')), pcd)
        return f'{file} point cloud created'

    @staticmethod
    def create_point_cloud_from_depth(depth_image, ply_dir, sub_dir, file):
        # Load a depth image
        # depth = o3d.io.read_image(depth_image)
        # Define camera intrinsics
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)

        # Create a point cloud from the depth image
        pcd = o3d.geometry.PointCloud.create_from_depth_image(
            depth=depth_image, intrinsic=intrinsic, depth_scale=1000.0, depth_trunc=1000.0, stride=1)

        # Flip it, otherwise the pointcloud will be upside down
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        o3d.visualization.draw_plotly([pcd], zoom=0.3412)
        # Save point cloud
        o3d.io.write_point_cloud(os.path.join(ply_dir, sub_dir, file.replace('.png', '_dpt.pcd')), pcd)
        return f'{file} point cloud created'

