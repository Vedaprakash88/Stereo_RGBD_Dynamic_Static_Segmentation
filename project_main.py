from rgbd2pointcloud import Rgb23D
import os
from tqdm import tqdm
import logging
from stereo2RGBD import load_calib
rgb_dir, depth_dir, ply_dir, sub_dirs, error_folder, error_df, calib_folder = Rgb23D.get_paths()

for sub_dir in sub_dirs:
    # Get list of files in each directory
    rgb_files = os.listdir(os.path.join(rgb_dir, sub_dir))
    depth_files = os.listdir(os.path.join(depth_dir, sub_dir))

    # Find common files in both directories
    common_files = list(set(rgb_files) & set(depth_files))

    for file in tqdm(iterable=common_files, desc=sub_dir + ' Point Clouds Created:', colour='blue', unit='image'):
        # Load RGB image
        try:
            rgb_image, depth_image = Rgb23D.read_images(rgb_dir, depth_dir, sub_dir, file)
            calib_file = os.path.join(calib_folder, sub_dir, file[:6] + '.txt')
            _, _, calibration_params = load_calib(calib_file)
            msg = Rgb23D.create_point_cloud(rgb_image, depth_image, ply_dir, sub_dir, file, calib_file)
            # msg2 = Rgb23D.create_point_cloud_from_depth(depth_image, ply_dir, sub_dir, file)
        except Exception as e:
            logging.error("Error occurred", exc_info=True)
            error_type = {'Error_Type': type(e).__name__, 'Explanation': str(e), 'Stage': 'Image Processing',
                          'File_Name': file}
            error_df = error_df._append(error_type, ignore_index=True)

# Save error DataFrame to Excel
if not error_df.dropna().empty:
    error_df.to_excel(os.path.join(error_folder, 'errors.xlsx'), index=False)
