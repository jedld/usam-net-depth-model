from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor
import cv2
import os
import numpy as np
import torch
import onnxruntime
import tqdm as tqdm
from onnxruntime.quantization import QuantType
from onnxruntime.quantization.quantize import quantize_dynamic

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

sam = sam_model_registry["vit_b"](checkpoint="tmp/sam_vit_b_01ec64.pth")
sam.to(device)

image_folders = [
                 'tmp/data/train/left_images',
                #  'tmp/data/train/right_images',
                #  'tmp/data/test/left_images',
                #  'tmp/data/test/right_images'
                ]

# os.makedirs('tmp/data/test/left_sky_masks', exist_ok=True)
# os.makedirs('tmp/data/test/right_masks', exist_ok=True)
os.makedirs('tmp/data/train/left_sky_masks', exist_ok=True)
# os.makedirs('tmp/data/train/right_masks', exist_ok=True)
predictor = SamPredictor(sam)

for image_folder in image_folders:
    print(f'Processing images in {image_folder}')
    for image_file in tqdm.tqdm(os.listdir(image_folder)):
        image_path = os.path.join(image_folder, image_file)
        mask_image_path = image_path.replace('left_images', 'left_sky_masks').replace('.jpg', '.png')
        if not os.path.exists(mask_image_path):
            if os.path.isfile(image_path):
                image = cv2.imread(image_path)

                # read corresponding depth map
                depth_map_path = image_path.replace('left_images', 'depth_maps').replace('.jpg', '.png')
                depth_map = cv2.imread(depth_map_path, cv2.IMREAD_GRAYSCALE)

                
                assert image.shape[:2] == depth_map.shape, f"Image and depth map shapes do not match: {image.shape} vs {depth_map.shape}"
                height, width = image.shape[:2]

                # select 16 random points along the top of the image
                x_coors = np.random.randint(0, width, (16))
                y_coors = np.random.randint(0, 5, (16))
                points = [(x, y) for x, y in zip(x_coors, y_coors)]

                # make sure the depth map values are zero at these points
                valid_points = []
                for x, y in points:
                    if depth_map[y, x] == 0:
                        valid_points.append((x, y))


                predictor.set_image(image)
                point_labels = [0] * len(points)
                point_labels = np.array(point_labels)
                points = np.array(points)
                masks, _, _ = predictor.predict(points, point_labels=point_labels)

                valid_pixels = (depth_map > 0).astype(np.uint8)

                converted_mask = np.zeros_like(valid_pixels)
                for index, mask in enumerate(masks):
                    for y in range(mask.shape[0]):
                        for x in range(mask.shape[1]):
                            if mask[y, x] > 0 and depth_map[y, x] == 0:
                                converted_mask[y, x] = 1

                # find the contours of the mask
                contours, _ = cv2.findContours(converted_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                mask = np.zeros_like(converted_mask)
                if len(contours) > 0:
                    # find the largest contour
                    largest_contour = max(contours, key=cv2.contourArea)

                    # draw the largest contour on the mask
                    
                    cv2.drawContours(mask, [largest_contour], -1, 255, -1)

                    # write the sky mask to disk
                    cv2.imwrite(mask_image_path, mask)
                else:
                    print("No contours found")
                    cv2.imwrite(mask_image_path, mask)
