import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet18
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from transforms import transform_disparity_fn, transform_fn, transform_seg_fn, test_transform_fn, test_transform_seg_fn
import cv2

import torch.nn as nn


class BaselineStereoCNN(nn.Module):
    def __init__(self, device):
        super(BaselineStereoCNN, self).__init__()
        self.down1 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64))
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128))
        self.down3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256)
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(512)
        )
        self.down5 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(1024)
        )


        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(512)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256)
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128)
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64)
        )

        self.up5 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=(4, 3), stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32)
        )

        self.conv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
        self.device = device

    def forward(self, x):
        down1 = self.down1(x)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)
        down5 = self.down5(down4)
        
        up1 = self.up1(down5) + down4
        up2 = self.up2(up1) + down3
        up3 = self.up3(up2) + down2
        up4 = self.up4(up3) + down1
        up5 = self.up5(up4)
        return self.conv(up5) * 255
        
    
    def inference(self, left_img, right_img):
        transform = test_transform_fn()
        left_img = transform(left_img).unsqueeze(0).to(self.device)
        right_img = transform(right_img).unsqueeze(0).to(self.device)
        input = torch.cat((left_img, right_img), 1)
        return self.forward(input), None
    
class BaselineStereoCNN2(nn.Module):
    def __init__(self, device):
        super(BaselineStereoCNN2, self).__init__()
        self.down1 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64))
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128))
        self.down3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256)
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(512)
        )
        self.down5 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(1024)
        )


        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(512)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256)
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128)
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64)
        )

        self.up5 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=(4, 3), stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32)
        )

        self.conv = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(128, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
        self.device = device

    def forward(self, x):
        down1 = self.down1(x)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)
        down5 = self.down5(down4)
        
        up1 = self.up1(down5) + down4
        up2 = self.up2(up1) + down3
        up3 = self.up3(up2) + down2
        up4 = self.up4(up3) + down1
        up5 = self.up5(up4)
        return self.conv(up5) * 255
        
    
    def inference(self, left_img, right_img):
        transform = test_transform_fn()
        left_img = transform(left_img).unsqueeze(0).to(self.device)
        right_img = transform(right_img).unsqueeze(0).to(self.device)
        input = torch.cat((left_img, right_img), 1)
        return self.forward(input), None
    
class BaseSegmentationCNN(nn.Module):
    def inference(self, left_img, right_img):
        transform = test_transform_fn()
        transform_segmentation = test_transform_seg_fn()
        
        # check if there is a batch dimension, remove it
        if len(left_img.shape) == 4:
            left_img = left_img[0]
            right_img = right_img[0]

        left_mask = self.generate_segment_map(left_img)
        left_img = transform(left_img).to(self.device).unsqueeze(0)
        right_img = transform(right_img).to(self.device).unsqueeze(0)
        
        
        left_mask = transform_segmentation(left_mask).to(self.device).unsqueeze(0)
        input = torch.cat((left_img, right_img, left_mask), 1)
        return self.forward(input), left_mask


    def generate_segment_map(self, image):
        masks = self.mask_generator.generate(image)
        processed_image = np.zeros_like(image)

        for i, mask in enumerate(masks):
            color = (i%255, i*10%255, i*100%255)
            color_mask = np.zeros_like(image)
            color_mask[ : , : , : ] = color
            segmentation_mask = np.array(mask['segmentation'], dtype=np.uint8)
            processed_image += cv2.bitwise_and(color_mask, color_mask, mask=segmentation_mask)
            
        return processed_image

class SegStereoCNN2(BaseSegmentationCNN):
    def __init__(self, device, load_sam=False):
        super(SegStereoCNN2, self).__init__()
        self.down1 = nn.Sequential(
            nn.Conv2d(9, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64))
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128))
        self.down3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256)
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(512)
        )
        self.down5 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(1024)
        )
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(512)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256)
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128)
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64)
        )
        self.up5 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=(4, 3), stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32)
        )
        self.conv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
        
        self.device = device
        if load_sam:
            sam = sam_model_registry["vit_b"](checkpoint="tmp/sam_vit_b_01ec64.pth")
            sam.to(device)
            self.mask_generator = SamAutomaticMaskGenerator(sam)

    def forward(self, x):
        down1 = self.down1(x)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)
        down5 = self.down5(down4)
        
        up1 = self.up1(down5) + down4
        up2 = self.up2(up1) + down3
        up3 = self.up3(up2) + down2
        up4 = self.up4(up3) + down1
        up5 = self.up5(up4)
        return self.conv(up5) * 255

class SegStereoCNN(BaseSegmentationCNN):
    def __init__(self, device, load_sam=False):
        super(SegStereoCNN, self).__init__()
        self.down1 = nn.Sequential(
            nn.Conv2d(9, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64))
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128))
        self.down3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256)
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(512)
        )
        self.down5 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(1024)
        )
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(512)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256)
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128)
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64)
        )
        self.up5 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=(4, 3), stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32)
        )
        self.conv = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(128, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
        self.device = device
        if load_sam:
            sam = sam_model_registry["vit_b"](checkpoint="tmp/sam_vit_b_01ec64.pth")
            sam.to(device)
            self.mask_generator = SamAutomaticMaskGenerator(sam)

    def forward(self, x):
        down1 = self.down1(x)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)
        down5 = self.down5(down4)
        
        up1 = self.up1(down5) + down4
        up2 = self.up2(up1) + down3
        up3 = self.up3(up2) + down2
        up4 = self.up4(up3) + down1
        up5 = self.up5(up4)
        return self.conv(up5) * 255

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.softmax = nn.Softmax(dim=-1)  # Softmax over the last dimension to create attention maps

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, width * height)
        value = self.value_conv(x).view(batch_size, -1, width * height)

        attention = self.softmax(torch.bmm(query, key))
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        
        return out + x  # Skip connection

class SASegStereoCNN(BaseSegmentationCNN):
    def __init__(self, device, load_sam=False):
        super(SASegStereoCNN, self).__init__()
        self.down1 = nn.Sequential(
            nn.Conv2d(9, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64))
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128))
        self.down3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256)
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(512)
        )
        self.down5 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(1024)
        )
        
        self.self_attention = SelfAttention(1024)

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(512)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256)
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128)
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64)
        )
        self.up5 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=(4, 3), stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32)
        )
        self.conv = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(128, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
        self.device = device
        if load_sam:
            sam = sam_model_registry["vit_b"](checkpoint="tmp/sam_vit_b_01ec64.pth")
            sam.to(device)
            self.mask_generator = SamAutomaticMaskGenerator(sam)

    def forward(self, x):
        down1 = self.down1(x)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)
        down5 = self.down5(down4)
        sa = self.self_attention(down5)    
        up1 = self.up1(sa) + down4
        up2 = self.up2(up1) + down3
        up3 = self.up3(up2) + down2
        up4 = self.up4(up3) + down1
        up5 = self.up5(up4)
        return self.conv(up5) * 255
    

class SASegStereoCNN2(BaseSegmentationCNN):
    def __init__(self, device, load_sam=False):
        super(SASegStereoCNN2, self).__init__()
        self.down1 = nn.Sequential(
            nn.Conv2d(9, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64))
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128))
        self.down3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256)
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(512)
        )
        self.down5 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(1024)
        )
        
        self.self_attention = SelfAttention(1024)

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(512)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256)
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128)
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64)
        )
        self.up5 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=(4, 3), stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32)
        )
        self.conv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
        self.device = device
        if load_sam:
            sam = sam_model_registry["vit_b"](checkpoint="tmp/sam_vit_b_01ec64.pth")
            sam.to(device)
            self.mask_generator = SamAutomaticMaskGenerator(sam)

    def forward(self, x):
        down1 = self.down1(x)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)
        down5 = self.down5(down4)
        sa = self.self_attention(down5)    
        up1 = self.up1(sa) + down4
        up2 = self.up2(up1) + down3
        up3 = self.up3(up2) + down2
        up4 = self.up4(up3) + down1
        up5 = self.up5(up4)
        return self.conv(up5) * 255


class SASegMonoCNN(BaseSegmentationCNN):
    def __init__(self, device, load_sam=False):
        super(SASegMonoCNN, self).__init__()
        self.down1 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64))
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128))
        self.down3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256)
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(512)
        )
        self.down5 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(1024)
        )
        
        self.self_attention = SelfAttention(1024)

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(512)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256)
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128)
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64)
        )
        self.up5 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=(4, 3), stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32)
        )
        self.conv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
        self.device = device
        if load_sam:
            sam = sam_model_registry["vit_b"](checkpoint="tmp/sam_vit_b_01ec64.pth")
            sam.to(device)
            self.mask_generator = SamAutomaticMaskGenerator(sam)

    def forward(self, x):
        down1 = self.down1(x)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)
        down5 = self.down5(down4)
        sa = self.self_attention(down5)    
        up1 = self.up1(sa) + down4
        up2 = self.up2(up1) + down3
        up3 = self.up3(up2) + down2
        up4 = self.up4(up3) + down1
        up5 = self.up5(up4)
        return self.conv(up5) * 255

class SAStereoCNN3(BaselineStereoCNN):
    def __init__(self, device):
        super(SAStereoCNN3, self).__init__(device)
        #  input channels = 6, 400x879
        self.down1 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64))
        # 64, 200x440
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128))
        # 128, 100x220
        self.down3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256)
        )
        # 256, 50x110
        self.down4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(512)
        )
        # 512, 25x55
        self.down5 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(1024)
        )
        
        self.self_attention = SelfAttention(1024)

        # 1024, 13x28
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(512)
        )

        # 512, 25x55
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256)
        )

        # 256, 50x110
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128)
        )

        # 128, 100x220
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64)
        )

        # 64, 200x440
        self.up5 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=(4, 3), stride=2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32)
        )

        # 32, 400x879
        self.conv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=4, stride=1, padding=3),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
        self.device = device


    def forward(self, x):
        down1 = self.down1(x)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)
        down5 = self.down5(down4)
        sa = self.self_attention(down5)    
        up1 = self.up1(sa) + down4
        up2 = self.up2(up1) + down3
        up3 = self.up3(up2) + down2
        up4 = self.up4(up3) + down1
        up5 = self.up5(up4)
        return self.conv(up5) * 255    
    
