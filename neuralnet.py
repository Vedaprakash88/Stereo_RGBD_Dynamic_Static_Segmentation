# Building my own NN.
import open3d as o3d
source = o3d.io.read_point_cloud("D:\\10. SRH_Academia\\1. All_Notes\\4. Thesis\\5. WIP\\Data\\KITTI_Motion\\data_scene_flow\\PCD\\training\\000000_10.pcd")
target = o3d.io.read_point_cloud("D:\\10. SRH_Academia\\1. All_Notes\\4. Thesis\\5. WIP\\Data\\KITTI_Motion\\data_scene_flow\\PCD\\training\\000000_11.pcd")
# aligned = o3d.io.read_point_cloud("D:\\10. SRH_Academia\\1. All_Notes\\4. Thesis\\5. WIP\\Data\\KITTI_Motion\\data_scene_flow\\aligned\\training\\000100.ply")

o3d.visualization.draw_geometries([source])
o3d.visualization.draw_geometries([target])
# o3d.visualization.draw_geometries([aligned])