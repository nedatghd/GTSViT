# Hyperspectral Image Classification with GTS-ViT

This repository contains the implementation of an end-to-end framework for hyperspectral image classification based on **MSDA —Multi-Scale Dilated Attention Module** and **Transformer with Groupwise Token Selective Self-Attention (GTSSA)**. The model extracts both neighborhood information and spatial relational information for classifying hyperspectral images with high precision.

![](https://i0.wp.com/eos.org/wp-content/uploads/2021/07/remote-imaging-spectrometers-data-cube.png?fit=820%2C615&ssl=1)


## [Datasets](#datasets)

- [Indian Pines Dataset](https://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes#Indian_Pines)
- [Salinas scene](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes)
- [WHU-Hi](https://rsidea.whu.edu.cn/resource_WHUHi_sharing.htm)

## Run the repo

To run this project, follow these steps:

1. Clone the repository:

    ```shell
    git clone https://github.com/nedatghd/GTSViT.git
    cd GTSViT
    ```

2. Install the required dependencies.

4. Populate your `data` folder with datasets

4. Run 

    ```shell
    python main.py --model gtsvit --dataset_name ip --dataset_dir ./datasets --patch_size 8 --num_run 10 --epoch 200 --bs 128 --ratio 0.10 --wandb_project GSCViT-HSI-ip
    ```
