import os
import zipfile
import shutil
from tqdm import tqdm

folder_names = ["left_images", "right_images", "disparity_maps", "depth_maps"]

def recursive_extract_zip(zip_file_or_folder, target_folder):
    if os.path.isdir(zip_file_or_folder):
        print(f'checking {zip_file_or_folder}')
        for file in os.listdir(zip_file_or_folder):
            file_path = os.path.join(zip_file_or_folder, file)
            if file.endswith('.zip'):
                recursive_extract_zip(file_path, target_folder)
                os.remove(file_path)
            elif os.path.isdir(file_path):
                recursive_extract_zip(file_path, target_folder)
            else:
                file_basename = os.path.basename(file)
                target_filename = os.path.join(target_folder, file_basename)
                if not os.path.exists(target_filename):
                    shutil.move(file_path, target_folder)
                else:
                    print(f'File {target_filename} already exists, skipping')
        shutil.rmtree(zip_file_or_folder)
    else:
        with zipfile.ZipFile(zip_file_or_folder, 'r') as zip_ref:
            tmp_subfolder = f"{zip_file_or_folder}_tmp"
            tmp_subfolder = os.path.join(target_folder, tmp_subfolder)
            zip_ref.extractall(tmp_subfolder)
            recursive_extract_zip(tmp_subfolder, target_folder)

def extract_and_sort_files(source_folder, target_folder_dict):
    def filename_is_in_keys(filename, target_folder_dict):
        for keyword in target_folder_dict.keys():
            if keyword in filename:
                return target_folder_dict[keyword]
        return False

    # List all files in the source folder
    files = os.listdir(source_folder)
    
    os.makedirs("tmp", exist_ok=True)
    for file in files:
        # Construct full file path
        file_path = os.path.join(source_folder, file)
        
        # Check if the file is a zip file
        if file.endswith('.zip') and filename_is_in_keys(file, target_folder_dict):
            target_folder = filename_is_in_keys(file, target_folder_dict)
            tmp_folder = f"tmp/{target_folder}" 
            os.makedirs(tmp_folder, exist_ok=True)
            # Open the zip file

            print(f'Extracting {file} to {tmp_folder}')
            recursive_extract_zip(file_path, tmp_folder)

# Example usage
source_folder = 'data/downloaded'
target_folder_dict = {
    'train-depth-map': 'data/train/depth_maps',
    'train-left-image': 'data/train/left_images',
    'train-right-image': 'data/train/right_images',
    'train-disparity-map': 'data/train/disparity_maps',
    'test-depth-map': 'data/test/depth_maps',
    'test-left-image': 'data/test/left_images',
    'test-right-image': 'data/test/right_images',
    'test-disparity-map': 'data/test/disparity_maps',
    'calib': 'data/calibration'
}

extract_and_sort_files(source_folder, target_folder_dict)


# Analyze the number of images, and the size of the images
import cv2
import numpy as np

def analyze_images(folder):
    files = os.listdir(folder)
    print(f'Analyzing {len(files)} images in {folder}')
    # verify each image if they are valid
    for file in tqdm(files):
        file_path = os.path.join(folder, file)
        img = cv2.imread(file_path)
        if img is None:
            if os.path.isdir(file_path):
                # remove empty folder including subdirectories
                print(f'Removing empty folder: {file_path}')
                shutil.rmtree(file_path)
            else:
                print(f'Invalid image: {file_path}')
                raise Exception(f'Invalid image: {file_path}')
        elif img.dtype != np.uint8:
            print(f'Invalid image type: {file_path} with dtype {img.dtype}')
            continue
    print(f'All images are valid {len(files)} images in {folder}')
    return len(files)


        

#analyze train images
analyze_images('tmp/data/train/left_images')
analyze_images('tmp/data/train/right_images')
analyze_images('tmp/data/train/disparity_maps')
analyze_images('tmp/data/train/depth_maps')

#analyze test images
analyze_images('tmp/data/test/left_images')
analyze_images('tmp/data/test/right_images')
analyze_images('tmp/data/test/disparity_maps')
analyze_images('tmp/data/test/depth_maps')
