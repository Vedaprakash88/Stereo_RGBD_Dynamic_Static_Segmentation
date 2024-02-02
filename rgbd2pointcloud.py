import os
import pandas as pd
import open3d as o3d
from tqdm import tqdm
import logging
import numpy as np
import matplotlib.pyplot as plt


def load_calib(calib_file):
    with open(calib_file, 'r') as f:
        fin = f.readlines()
        for line in fin:
            if line[:9] == 'P_rect_02':
               cam2_parameters = [line[11:23], line[76:89], line[37:49], line[89:102]]
    return cam2_parameters


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


def visualize_ply(file_path):
    # Load the .ply file
    ply = o3d.io.read_point_cloud(file_path)

    # Visualize the point cloud
    o3d.visualization.draw_geometries([ply])


def read_images(rgb_dir, depth_dir, sub_dir, file):
    # Get list of files in each directory
    rgb_image = o3d.io.read_image(os.path.join(rgb_dir, sub_dir, file))

    # Load depth image
    depth_image = o3d.io.read_image(os.path.join(depth_dir, sub_dir, file))

    return rgb_image, depth_image


def create_point_cloud(rgb_image, depth_image, ply_dir, sub_dir, file, calibration_params):
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_image, depth_image,
                                                                    convert_rgb_to_intensity=True)

    calibration_params = list(map(float, calibration_params))

    fx = calibration_params[0]  # Focal length in x-axis
    fy = calibration_params[1]  # Focal length in y-axis
    cx = calibration_params[2]  # Principal point in x-axis
    cy = calibration_params[3]  # Principal point in y-axis

    # Get the width and height of the image
    height, width = np.asarray(rgb_image).shape[:2]

    # Create a PinholeCameraIntrinsic object using the intrinsic parameters
    intrinsic = o3d.pybind.camera.PinholeCameraIntrinsic() #width=width, height=height, fx=fx, fy=fy, cx=cx, cy=cy)

    # Create point cloud
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)

    # Flip it, otherwise the pointcloud will be upside down
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    # o3d.visualization.draw_plotly([pcd], zoom=0.3412)

    # Save point cloud
    o3d.io.write_point_cloud(os.path.join(ply_dir, sub_dir, file.replace('.png', '.pcd')), pcd)
    return f'{file} point cloud created'


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


rgb_dir, depth_dir, ply_dir, sub_dirs, error_folder, error_df, calib_folder = get_paths()

for sub_dir in sub_dirs:
    # Get list of files in each directory
    rgb_files = os.listdir(os.path.join(rgb_dir, sub_dir))
    depth_files = os.listdir(os.path.join(depth_dir, sub_dir))

    # Find common files in both directories
    common_files = list(set(rgb_files) & set(depth_files))

    for file in tqdm(iterable=common_files, desc=sub_dir + ' Point Clouds Created:', colour='blue', unit='image'):
        # Load RGB image
        try:
            rgb_image, depth_image = read_images(rgb_dir, depth_dir, sub_dir, file)
            calib_file = os.path.join(calib_folder, sub_dir, file[:6] + '.txt')
            calibration_params = load_calib(calib_file)
            msg = create_point_cloud(rgb_image, depth_image, ply_dir, sub_dir, file, calibration_params)
            # msg2 = Rgb23D.create_point_cloud_from_depth(depth_image, ply_dir, sub_dir, file)
        except Exception as e:
            logging.error("Error occurred", exc_info=True)
            error_type = {'Error_Type': type(e).__name__, 'Explanation': str(e), 'Stage': 'Image Processing',
                          'File_Name': file}
            error_df = error_df._append(error_type, ignore_index=True)

# Save error DataFrame to Excel
if not error_df.dropna().empty:
    error_df.to_excel(os.path.join(error_folder, 'errors.xlsx'), index=False)
