# USAM-Net

## Description

This project aims to predict the disparity maps using stereo images. Disparity maps provide information about the depth of objects in a scene, which can be useful for various computer vision applications such as 3D reconstruction, object detection, and autonomous navigation. The USAM-Net model uses segmentation maps as an additional feature that can be used to predict the disparity.

## Features

- Input: Stereo images (left and right), Segmentation Map for Left Image using Segment Anything Model from Facebook
- Output: Disparity map
- Algorithm: Custom U-Net like architecture with batch normalizatoin and an attention layer
- Evaluation: GD, ARD, D1 and EPE loss metrics

## Datasets

This project uses the following datasets:

1. "DrivingStereo: A Large-Scale Dataset for Stereo Matching in Autonomous Driving Scenarios" - https://drivingstereo-dataset.github.io/

## Installation

1. Clone the repository:

    ```bash
    git clone git@github.com:jedld/usam-net-depth-model.git
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. The following stereo dataset is used for training, dataset needs to be downloaded first. Place the zip files (from google drive) to the tmp/data folder. The images should be organized in the following manner:

```plaintext
tmp
├── data
    ├── calib
    ├── test
    │   ├── depth_maps
    │   ├── disparity_maps
    │   ├── left_images
    │   ├── left_masks
    │   ├── right_images
    │   └── right_masks
    └── train
        ├── depth_maps
        ├── disparity_maps
        ├── left_images
        ├── left_masks
        ├── left_sky_masks
        ├── right_images
        └── right_masks
```

4. You will need to generate the segment masks from the Segment Anything Model. Please see documentation at
https://github.com/facebookresearch/segment-anything on how to setup segment anything. You need to download the Vit-B SAM model to the project folder. After that you can run the generate_segment_maps.py script to
generate the segment masks.

## Usage

A Jupyter notebook is included to run the training as well as experiments:

benchmark.ipynb

## Reproducing results

The following scripts can be used to train the models from scratch.

baseline
```bash
python train_baseline.py --data=driving_stereo/ --wandb=true --batch-size=32 --num-epochs=30
```

segmentation
```bash
python train_seg2.py --data=driving_stereo/ --wandb=true --batch-size=32 --num-epochs=30
```

segmentation+attention

```bash
python train_seg_sa2.py --data=driving_stereo/ --wandb=true --batch-size=32 --num-epochs=30
```

## Results

### Performance Scores of Machine Learning Models on the DrivingStereo Dataset

| Model Type                                  | EPE   | D1 Error | GD     | L1-loss |
|---------------------------------------------|-------|----------|--------|---------|
| U-Net baseline (OURS)                       | 0.964 | 2.94%    | 4.17%  | 0.12    |
| USAM-Net (Segmentation) (OURS)              | 0.924 | 2.65%    | 3.68%  | 0.115   |
| USAM-Net (Segmentation+Attention) (OURS)    | 0.928 | 2.95%    | 3.73%  | 0.116   |
| CFNet                                       | 0.98  | 1.46%    | -      | -       |
| SegStereo                                   | 1.32  | 5.89%    | 4.78%  | -       |
| EdgeStereo                                  | 1.19  | 3.47%    | 4.17%  | -       |
| iResNet                                     | 1.24  | 4.27%    | 4.23%  | -       |


## Contributing

Contributions are welcome! If you would like to contribute to this project, please follow the guidelines in [CONTRIBUTING.md](CONTRIBUTING.md).

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

For any questions or inquiries, please contact [your-email@example.com](mailto:your-email@example.com).