import os
import cv2
import numpy as np
import logging
import pandas as pd
from tqdm import tqdm
import open3d as o3d

def triangulation_stereo(disparity, P1, P2):

    # Get 3D coordinates using triangulation
    points_2d = np.nonzero(disparity)
    points_2d = np.vstack((points_2d[1], points_2d[0]))  # Swap x and y
    points_2d_homogeneous = np.vstack((points_2d, np.ones((1, points_2d.shape[1]))))

    # Split into two sets of points
    points_set1 = points_2d_homogeneous[:, :points_2d_homogeneous.shape[1] // 2]
    points_set2 = points_2d_homogeneous[:, points_2d_homogeneous.shape[1] // 2:]

    # Triangulate points
    points_3d_homogeneous = cv2.triangulatePoints(P1, P2, points_set1, points_set2)

    # Convert homogeneous 3D coordinates to 3D coordinates
    points_3D = (points_3d_homogeneous[:-1, :] / points_3d_homogeneous[-1, :]).T

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3D.reshape(-1, 3))
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    # reflect on x-axis
    points_3D[:, :, 0] = -points_3D[:, :, 0]

    return points_3D, pcd


# Define the paths
# Normalize data
def normalize_images(*args):
    results = []
    for img in args:
        result = np.apply_along_axis(lambda x: (x / 255.0), 0, img)
        results.append(result)
    return tuple(results)


# Function to load calibration file
def load_calib(calib_file):
    with open(calib_file, 'r') as f:
        fin = f.readlines()
        for line in fin:
            if line[:9] == 'P_rect_02':
                P2 = np.array(line[11:].strip().split(" ")).astype('float32').reshape(3, -1)
            elif line[:9] == 'P_rect_03':
                P3 = np.array(line[11:].strip().split(" ")).astype('float32').reshape(3, -1)
    return P2, P3


# Function to rectify images
def stereo_rectify(img_clr, P2, P3):
    cam1 = P2[:, :3]  # left image - P2
    cam2 = P3[:, :3]  # right image - P3

    Tmat = np.array([0.54, 0., 0.])  # from image of KITTI car

    rev_proj_matrix = np.zeros((4, 4))
    rev_proj_matrix[:3, 3] = np.asarray([695.7519, 559.7571, 903.7596])
    x = -1/(-0.4731050)
    y = (695.7519 - 696.0217) / -0.4731050
    rev_proj_matrix[3:, 2:] = np.asarray([x, y])

    # R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify()

    cv2.stereoRectify(cameraMatrix1=cam1, cameraMatrix2=cam2, distCoeffs1=0, distCoeffs2=0,
                      imageSize=img_clr.shape[:2], R=np.identity(3), T=Tmat, R1=None,
                      R2=None, P1=None, P2=None, Q=rev_proj_matrix)
    return rev_proj_matrix


# Function to generate disparity map
def generate_disparity_map(left_img, right_img):
    stereo = cv2.StereoSGBM_create(numDisparities=16, blockSize=5)
    disparity = stereo.compute(left_img, right_img).astype(np.float32) / 16.0
    return disparity

# Function to generate RGB-D image
def generate_rgbd_image(points_3D):
    # Create RGB-D image
    depth_image = points_3D[..., 2]  # Depth values
    depth_image = points_3D[..., 2]  # Depth values
    return depth_image


def get_3D(disparity, rev_proj_matrix):
    points_3D = cv2.reprojectImageTo3D(disparity, rev_proj_matrix)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3D.reshape(-1, 3))
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    points_3D[:, :, 0] = -points_3D[:, :, 0]  # reflect on x-axis
    return points_3D, pcd


# Function to save disparity map and RGB-D image
def save_images(disp_output_folder, rgb_output_folder, depth_output_folder, disparity, left_img_clr, depth_image, pcd,
                img_file, subfolder):
    # get right folders

    disp_output_folder = os.path.join(disp_output_folder, subfolder)
    os.makedirs(disp_output_folder, exist_ok=True)
    rgb_output_folder = os.path.join(rgb_output_folder, subfolder)
    os.makedirs(rgb_output_folder, exist_ok=True)
    depth_output_folder = os.path.join(depth_output_folder, subfolder)
    os.makedirs(depth_output_folder, exist_ok=True)

    # Save disparity map
    disp_file = os.path.join(disp_output_folder, img_file)
    cv2.imwrite(filename=disp_file, img=disparity)

    # Save RGB image
    rgb_file = os.path.join(rgb_output_folder, img_file)
    cv2.imwrite(filename=rgb_file, img=left_img_clr)

    # Save Depth image
    depth_file = os.path.join(depth_output_folder, img_file)
    # depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8) # Norm_
    # alize depth to 0-255 and convert to 8-bit integer
    cv2.imwrite(filename=depth_file, img=depth_image)

    # save pcd
    cv2_pcd_folder = 'D:\\10. SRH_Academia\\1. All_Notes\\4. Thesis\\5. WIP\\Data\\KITTI_Motion\\data_scene_flow\\CV2_PCD'
    pcd_file = os.path.join(cv2_pcd_folder, subfolder, img_file.replace('.png', '.pcd'))
    o3d.io.write_point_cloud(pcd_file, pcd)

def first_do_this():
    base_path = "D:\\10. SRH_Academia\\1. All_Notes\\4. Thesis\\5. WIP\\Data\\KITTI_Motion\\"
    left_image_folder = os.path.join(base_path, "data_scene_flow\\image_2")
    right_image_folder = os.path.join(base_path, "data_scene_flow\\image_3")
    calib_folder = os.path.join(base_path, "data_scene_flow_calib\\calib_cam_to_cam")
    disp_output_folder = os.path.join(base_path, "data_scene_flow\Disp_images")
    rgb_output_folder = os.path.join(base_path, "data_scene_flow\\RGB_images")
    depth_output_folder = os.path.join(base_path, "data_scene_flow\\Depth_images")
    error_folder = os.path.join(base_path, "data_scene\\Error_Handling")
    error_df = pd.DataFrame()

    # Create output directories if they do not exist
    os.makedirs(disp_output_folder, exist_ok=True)
    os.makedirs(rgb_output_folder, exist_ok=True)
    os.makedirs(depth_output_folder, exist_ok=True)
    os.makedirs(error_folder, exist_ok=True)

    # Define the subfolders
    subfolders = ["testing", "training"]
    # Iterate over subfolders
    for subfolder in subfolders:
        left_images_path = os.path.join(left_image_folder, subfolder)
        right_images_path = os.path.join(right_image_folder, subfolder)
        calib_path = os.path.join(calib_folder, subfolder)

        # Get the list of image filenames in both directories
        left_images = os.listdir(left_images_path)
        right_images = os.listdir(right_images_path)

        # Find common files in both directories
        common_files = list(set(left_images) & set(right_images))

        # Iterate over images
        for img_file in tqdm(iterable=common_files, desc=f'Processing now: {subfolder}'):
            try:
                # Load images

                left_img_clr = cv2.imread(os.path.join(left_images_path, img_file))
                right_img_clr = cv2.imread(os.path.join(right_images_path, img_file))

                left_img_GS = cv2.imread(os.path.join(left_images_path, img_file), cv2.IMREAD_GRAYSCALE)
                right_img_GS = cv2.imread(os.path.join(right_images_path, img_file), cv2.IMREAD_GRAYSCALE)

                # Load calibration file
                calib_file = os.path.join(calib_path, img_file[:6] + '.txt')
                P2, P3 = load_calib(calib_file)

                # Generate disparity map
                disparity = generate_disparity_map(left_img_GS, right_img_GS)

                # generate depth using triangulation

                # points_3D, pcd = triangulation_stereo(disparity, P2, P3)

                # Rectify images
                rev_proj_matrix = stereo_rectify(right_img_clr, P2, P3)

                # Project to 3D
                points_3D, pcd = get_3D(disparity, rev_proj_matrix)

                # Generate RGB image and depth image
                depth_image = generate_rgbd_image(points_3D)

                # Save disparity map and RGB-D image
                save_images(disp_output_folder, rgb_output_folder, depth_output_folder, disparity, right_img_clr,
                            depth_image, pcd, img_file, subfolder)
            except cv2.error as e:
                logging.error("OpenCV Error occurred", exc_info=True)
                error_type = {'Error_Type': 'OpenCVError', 'Explanation': str(e), 'Stage': 'Image Processing',
                              'File_Name': img_file}
                error_df = error_df._append(error_type, ignore_index=True)
            except Exception as e:
                logging.error("Error occurred", exc_info=True)
                error_type = {'Error_Type': type(e).__name__, 'Explanation': str(e), 'Stage': 'Image Processing',
                              'File_Name': img_file}
                error_df = error_df._append(error_type, ignore_index=True)

    # Save error DataFrame to Excel
    if error_df.dropna().empty:
        error_df.to_excel(os.path.join(error_folder, 'errors.xlsx'), index=False)
