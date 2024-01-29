# Building my own NN.
import open3d as o3d
import cv2

source = o3d.io.read_point_cloud("D:\\10. SRH_Academia\\1. All_Notes\\4. Thesis\\5. WIP\\Data\\KITTI_Motion\\data_scene_flow\\PCD\\training\\000000_10.pcd")
target = o3d.io.read_point_cloud("D:\\10. SRH_Academia\\1. All_Notes\\4. Thesis\\5. WIP\\Data\\KITTI_Motion\\data_scene_flow\\PCD\\training\\000000_11.pcd")
# aligned = o3d.io.read_point_cloud("D:\\10. SRH_Academia\\1. All_Notes\\4. Thesis\\5. WIP\\Data\\KITTI_Motion\\data_scene_flow\\aligned\\training\\000100.ply")
x = cv2.imread("D:\\10. SRH_Academia\\1. All_Notes\\4. Thesis\\5. WIP\\Data\\KITTI_Motion\\data_scene_flow\\Depth_images\\training\\000106_10.png", cv2.IMREAD_GRAYSCALE)
print(x.shape)

o3d.visualization.draw_plotly([source], zoom=0.3412)
o3d.visualization.draw_plotly([target], zoom=0.3412)
# o3d.visualization.draw_geometries([aligned])