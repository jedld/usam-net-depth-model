import os
import torch
from torch.utils.data import DataLoader

import cv2
import numpy as np
from sa_model import BaselineStereoCNN
from tqdm import tqdm
from torchsummary import summary
import argparse
from dataset_kitti import StereoDataset
from metrics import calculate_ard_and_gd
from transforms import transform_disparity_fn, transform_fn, test_transform_seg_fn, test_transform_fn
import wandb


# ARD and GD parameters, set to the same as the DrivingStereo paper
min_depth = 0 
max_depth = 80 
r = 4 
interval = 8

# parse arguments to obtain location of the dataset
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='Location of the dataset')
parser.add_argument('--wandb', type=bool, help='Use wandb')
parser.add_argument('--num-epochs', type=int, default=30, help='Number of epochs')
parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
parser.add_argument('--num-workers', type=int, default=16, help='Number of workers')
parser.add_argument('--aa-bottomslice', type=bool, default=False, help='Use aa-bottom-slice')
parser.add_argument('--sky-mask', type=bool, default=False, help='Use the sky mask')
parser.add_argument('--finetune', type=bool, default=False, help='Finetune mode')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--weight-decay', type=float, default=0.0001, help='Weight decay')

args = parser.parse_args()

if args.data is None:
    print('Please provide the location of the dataset using --data')
    exit()

data_set_folder = args.data

BATCH_SIZE = args.batch_size
NUM_WORKERS = args.num_workers
num_epochs = args.num_epochs


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load transforms
transform = transform_fn()
transform_disparity = transform_disparity_fn()
test_transform = test_transform_fn()


model = BaselineStereoCNN(device)
model.to(device)

summary(model, (6, 400, 879))

project_name = 'stereo_cnn_baseline_ktti'

if args.finetune and os.path.exists(f"stereo_cnn_baseline_checkpoint2.checkpoint"):
    print("loading existing model for finetune...")
    model.load_state_dict(torch.load(f"stereo_cnn_baseline_checkpoint2.checkpoint"))

# load the train images and disparity maps
dataset = StereoDataset(f'{data_set_folder}/training/image_2',
                        f'{data_set_folder}/training/image_3',
                        f'{data_set_folder}/training/disp_noc_0',
                        transform=transform,
                        transform_disparity=transform_disparity,
                        )
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

def evaluate(model, dataloader):
    model.eval()
    with torch.no_grad():
        total_test_loss = 0
        total_gd = 0
        total_ard_values = 0
        for left_image, right_image, disparity_map in tqdm(dataloader):
            left_image = left_image.to(device)
            right_image = right_image.to(device)
            disparity_map = disparity_map.to(device)
            
            outputs = model(torch.cat((left_image, right_image), 1))
            
            mask = disparity_map > (0 + 1e-8)
            outputs = outputs * mask.float()

            test_loss = criterion(outputs, disparity_map)
            
            total_test_loss += test_loss.item()
            
    return total_test_loss/len(dataloader), total_ard_values/len(dataloader), total_gd/len(dataloader)

# define the loss function and the optimizer
criterion = torch.nn.SmoothL1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

if args.wandb:
    wandb.init(project='stereo-cnn-ktti', name=project_name, config=args)

best_loss = 100000
best_gd = 100000
best_epoch = 0

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for left_image, right_image, disparity_map in tqdm(dataloader):
        left_image = left_image.to(device)
        right_image = right_image.to(device)
        disparity_map = disparity_map.to(device)


        outputs = model(torch.cat((left_image, right_image), 1))
        

        mask = disparity_map > 1e-8
        outputs = outputs * mask.float()

        loss = criterion(outputs, disparity_map)
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    lr_scheduler.step()
    
    torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict()}, f'stereo_cnn_{project_name}_checkpoint.pth')
    torch.save(model.state_dict(), f'stereo_cnn_{project_name}_checkpoint_{epoch}.pth')
    torch.save(model.state_dict(), f'stereo_cnn_{project_name}.pth')
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader)}')
    
