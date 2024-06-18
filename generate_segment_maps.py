from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
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

vit_checkpoint = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"

# download the checkpoint to  tmp
# !wget -O tmp/sam_vit_h_4b8939.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
# !wget -O tmp/sam_vit_l_0b3195.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
# !wget -O tmp/sam_vit_b_01ec64.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

sam = sam_model_registry["vit_b"](checkpoint="tmp/sam_vit_b_01ec64.pth")
sam.to(device)
mask_generator = SamAutomaticMaskGenerator(sam)

vit_checkpoint = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"

# download the checkpoint to  tmp
# !wget -O tmp/sam_vit_h_4b8939.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth


image_folders = [
                 'tmp/data/train/left_images',
                #  'tmp/data/train/right_images',
                 'tmp/data/test/left_images',
                #  'tmp/data/test/right_images'
                ]

os.makedirs('tmp/data/test/left_masks', exist_ok=True)
# os.makedirs('tmp/data/test/right_masks', exist_ok=True)
os.makedirs('tmp/data/train/left_masks', exist_ok=True)
# os.makedirs('tmp/data/train/right_masks', exist_ok=True)

for image_folder in image_folders:
    print(f'Processing images in {image_folder}')
    for image_file in tqdm.tqdm(os.listdir(image_folder)):
        image_path = os.path.join(image_folder, image_file)
        mask_image_path = image_path.replace('left_images', 'left_masks').replace('right_images', 'right_masks')

        if not os.path.exists(mask_image_path):
            if os.path.isfile(image_path):
                image = cv2.imread(image_path)
                try:
                    masks = mask_generator.generate(image)
                except:
                    print(f'Error processing {image_path}')
                    continue
                processed_image = np.zeros_like(image)

                for i, mask in enumerate(masks):
                    color = (i%255, i*10%255, i*100%255)
                    color_mask = np.zeros_like(image)
                    color_mask[ : , : , : ] = color
                    segmentation_mask = np.array(mask['segmentation'], dtype=np.uint8)
                    processed_image += cv2.bitwise_and(color_mask, color_mask, mask=segmentation_mask)
                
                cv2.imwrite(mask_image_path, processed_image)
