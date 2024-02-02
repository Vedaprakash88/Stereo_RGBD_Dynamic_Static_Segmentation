# Building my own NN.
import numpy as np
import open3d as o3d
import cv2
import matplotlib.pyplot as plt

# source = o3d.io.read_point_cloud("D:\\10. SRH_Academia\\1. All_Notes\\4. Thesis\\5. WIP\\Data\\KITTI_Motion\\data_scene_flow\\PCD\\training\\000000_10.pcd")
# target = o3d.io.read_point_cloud("D:\\10. SRH_Academia\\1. All_Notes\\4. Thesis\\5. WIP\\Data\\KITTI_Motion\\data_scene_flow\\PCD\\training\\000000_11.pcd")
# # aligned = o3d.io.read_point_cloud("D:\\10. SRH_Academia\\1. All_Notes\\4. Thesis\\5. WIP\\Data\\KITTI_Motion\\data_scene_flow\\aligned\\training\\000100.ply")
#
ex10 = o3d.io.read_point_cloud("D:\\10. SRH_Academia\\1. All_Notes\\4. Thesis\\5. WIP\\Data\\KITTI_Motion\\data_scene_flow\\CV2_PCD\\training\\000100_10.pcd")
ex11 = o3d.io.read_point_cloud("D:\\10. SRH_Academia\\1. All_Notes\\4. Thesis\\5. WIP\\Data\\KITTI_Motion\\data_scene_flow\\CV2_PCD\\training\\000101_11.pcd")
# #
# #
o3d.visualization.draw_plotly([ex10], zoom=0.3412)
o3d.visualization.draw_plotly([ex11], zoom=0.3412)
o3d.visualization.draw_geometries([ex10])

# left = "D:\\10. SRH_Academia\\1. All_Notes\\4. Thesis\\5. WIP\\Data\\KITTI_Motion\\data_scene_flow\\image_2\\training\\000000_10.png"
# right = "D:\\10. SRH_Academia\\1. All_Notes\\4. Thesis\\5. WIP\\Data\\KITTI_Motion\\data_scene_flow\\image_3\\training\\000000_10.png"
# disp_old = "D:\\10. SRH_Academia\\1. All_Notes\\4. Thesis\\5. WIP\\Data\\KITTI_Motion\\data_scene_flow\\Disp_images\\training\\000000_10.png"
# def computeDepthMapSGBM(left, right, disp_old):
#     windowsize = 7
#     min_disp = 16
#     nDispFactor = 7
#     num_disp = 16*nDispFactor-min_disp
#
#     stereo = cv2.StereoSGBM.create(minDisparity=min_disp,
#                                    numDisparities=num_disp,
#                                    blockSize=windowsize,
#                                    P1=8*3*windowsize**2,
#                                    P2=32*3*windowsize**2,
#                                    disp12MaxDiff=1,
#                                    uniquenessRatio=15,
#                                    speckleWindowSize=0,
#                                    speckleRange=2,
#                                    preFilterCap=63,
#                                    mode=cv2.StereoSGBM_MODE_SGBM_3WAY)
#     disparity = stereo.compute(left, right).astype(np.float32) /16.0
#
#     print(disparity.shape)
#     print(disp_old.shape)
#
#     plt.imshow(disparity)
#     plt.colorbar()
#     plt.show()
#
#     plt.imshow(disp_old)
#     plt.colorbar()
#     plt.show()
#
# left2 = cv2.imread(left, cv2.IMREAD_GRAYSCALE)
# right2 = cv2.imread(right, cv2.IMREAD_GRAYSCALE)
# disp_old2 = cv2.imread(disp_old, cv2.IMREAD_GRAYSCALE)
#
# computeDepthMapSGBM(left2, right2, disp_old2)

# left_img_GS = cv2.blur(cv2.cvtColor(left_img_clr, cv2.COLOR_RGB2GRAY), (5, 5))
# right_img_GS = cv2.blur(cv2.cvtColor(right_img_clr, cv2.COLOR_RGB2GRAY), (5, 5))