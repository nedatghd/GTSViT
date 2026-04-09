# Hyperspectral Image Classification with Graph Convolutional Networks and Vision Transformers

This repository contains the implementation of an end-to-end framework for hyperspectral image classification based on **Graph Convolutional Networks (GCN)** and **Vision Transformers (ViT)**. The model extracts both neighborhood information and spatial relational information for classifying hyperspectral images with high precision.

![](https://i0.wp.com/eos.org/wp-content/uploads/2021/07/remote-imaging-spectrometers-data-cube.png?fit=820%2C615&ssl=1)

## [Abstract](#abstract)

Hyperspectral images contain information across the electromagnetic spectrum for every pixel in the remote sensing image. This allows the identification of various materials and objects with different spectral signatures at a high level of precision. Our proposed methodology combines the power of **Graph Convolutional Networks (GCN)** and **Vision Transformers (ViT)** to extract both neighborhood information and spatial relational information. We evaluated the performance of our method on two benchmark hyperspectral datasets, namely Indian Pines and Pavia University, and compared the achieved performance with state-of-the-art methods.

## [Methodology](#methodology)

### Data Splitting Methodology
For both the `Indian Pines` and `Pavia University` datasets, a careful data splitting methodology was employed to ensure unbiased training and testing sets. The process involves random sampling and a transductive method to prevent data leakage between the two sets.

**Splitting Ratio**: The pixels were divided into two disjoint sets with a split ratio of 30% for training and 70% for testing. This ratio was chosen to strike a balance between training data size and ensuring robust evaluation on the testing set.

**Uniform Distribution**: To maintain class label balance in both training and testing sets, random sampling was performed with a uniform distribution among class labels. This approach helps in preventing biases and ensures that the model is exposed to a representative set of examples from each class during training.

**Transductive Method**: A transductive approach to data splitting was employed to guarantee that there is no data leakage between training and testing pixels. This method ensures that the model generalizes well to unseen data by creating a clear separation between the two sets.

#### Data Generation Steps: 

The generation of training data follows a structured process:

1. **Sample with Replacement**: v pixels were sampled with replacement to introduce variability and augment the training dataset.

2.  **Graph Construction**: Compute the adjacency matrix (A), feature matrix (V), and edge matrix (E) for the graph representation.

3.  **Patch Storage**: Store the patches centered on the sampled pixels for subsequent use in training

## [Model](#model)
The proposed methodology for hyperspectral image semantic segmentation extracts both structured and unstructured data and makes use of mixed information retrieved by Graph Convolutional Networks (GCN) and Vision Transformers (ViT). The pixel-wise concatenation of GCN and ViT-based representations is fed into a simple linear layer, followed by a Softmax layer which transforms the logits into a probability distribution, to predict the semantic class.

The proposed model architecture is presented in the following figure:

<div align="center">
  <img src="./media/model_architecture.png" alt="Model Architecture" />
</div>


For detailed discussion on the methodology, refer to the [original paper](/media/IGARSS_2022_Hyperspectral.pdf).

## [Datasets](#datasets)

We used two benchmark datasets for hyperspectral image classification, namely Pavia University and Indian Pines datasets. 

1. **`Pavia University` dataset** consists of $612 \times 340$ pixels (9 classes), characterized by 113 spectral bands from 0.43 to 0.86 &mu;m with a spatial resolution of 1.3 meters. 

2. **`Indian Pines dataset`** consists of a multi-spectral image of $145 \times 145$ pixels (16 classes) with 224 spectral bands ranging from 0.4 to 2.5 &mu;m and a spatial resolution of 20 meters.


## [Results](#results)

We compared our combined GCN and ViT-based approach with other methods in the literature, in terms of overall accuracy (OA) and Kappa coefficient ($\kappa$). The results are shown in the table below:
<table align="center">
  <thead>
    <tr>
      <th>No.</th>
      <th>Method</th>
      <th>OA[%]</th>
      <th>$\kappa$</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="8"><b>Pavia University</b></td>
    </tr>
    <tr>
      <td><b>Proposed method</b></td>
      <td>99.10</td>
      <td>0.98</td>
    </tr>
    <tr>
      <td>SpectralFormer</td>
      <td>91.07</td>
      <td>0.88</td>
    </tr>
    <tr>
      <td>miniGCN</td>
      <td>79.79</td>
      <td>0.73</td>
    </tr>
    <tr>
      <td>FuNet-C</td>
      <td>92.20</td>
      <td>0.89</td>
    </tr>
    <tr>
      <td>Deep CNN</td>
      <td>0.94</td>
      <td>-</td>
    </tr>
    <tr>
      <td>SSAN</td>
      <td>98.02</td>
      <td>0.97</td>
    </tr>
    <tr>
      <td>3D-CNN-LR</td>
      <td>99.54</td>
      <td>0.99</td>
    </tr>
    <tr>
      <td rowspan="8"><b>Indian Pines</b></td>
    </tr>
    <tr>
      <td><b>Proposed method</b></td>
      <td>95.53</td>
      <td>0.94</td>
    </tr>
    <tr>
      <td>SpectralFormer</td>
      <td>81.76</td>
      <td>0.79</td>
    </tr>
    <tr>
      <td>miniGCN</td>
      <td>75.11</td>
      <td>0.71</td>
    </tr>
    <tr>
      <td>FuNet-C</td>
      <td>79.89</td>
      <td>0.77</td>
    </tr>
    <tr>
      <td>Deep CNN</td>
      <td>0.92</td>
      <td>-</td>
    </tr>
    <tr>
      <td>SSAN</td>
      <td>95.49</td>
      <td>0.94</td>
    </tr>
    <tr>
      <td>3D-CNN-LR</td>
      <td>97.56</td>
      <td>0.97</td>
    </tr>
  </tbody>
</table>

#### Params used

<table align="center">
  <tr>
    <th>Parameter</th>
    <th>Value</th>
  </tr>
  <tr>
    <td>batch_size</td>
    <td>64</td>
  </tr>
  <tr>
    <td>lr</td>
    <td>0.0001</td>
  </tr>
  <tr>
    <td>nfeat</td>
    <td>200</td>
  </tr>
  <tr>
    <td>nhid</td>
    <td>128</td>
  </tr>
  <tr>
    <td>nclass</td>
    <td>16</td>
  </tr>
  <tr>
    <td>depth</td>
    <td>3</td>
  </tr>
  <tr>
    <td>image_size</td>
    <td>200</td>
  </tr>
  <tr>
    <td>patch_size</td>
    <td>5</td>
  </tr>
  <tr>
    <td>dropout</td>
    <td>0.2</td>
  </tr>
  <tr>
    <td>heads</td>
    <td>8</td>
  </tr>
</table>



For a visual evaluation of the results, we present an inference on `Pavia University Dataset` given the ground truth labels.

<div align="center">
  <img src="./media/data_sample.png" alt="Data Sample" />
</div>

*Inference on Pavia University Dataset: Predicted labels overlaid on ground truth labels.*





## [Confusion Matrices](#confusion-matrices)

#### Pavia University
<div align="center">
  <img src="./media/pavia_confusion.png" alt="Pavia University Results" />
</div>

#### Indian Pines
<div align="center">
  <img src="./media/indian_pines_confusion.png" alt="Indian Pines Results" />
</div>


## [Conclusion](#conclusion)
Our approach for hyperspectral image classification reached an overall accuracy of 99.10 \% on `Pavia University` and 95.53\% on `Indian Pines` datasets. Using simple, but powerful architectures, the results are in line with the best performing architectures in the literature.

The proposed GCN and ViT-based approach achieves competitive results in hyperspectral image classification. By combining the strengths of graph neural networks and vision transformers, the method demonstrates effectiveness in extracting both spatial and spectral features. The results on `Pavia University` and `Indian Pines` datasets are comparable to the state-of-the-art. Additionally, a novel data generation method for training sets using random sampling with replacement is introduced, contributing to performance improvement for small datasets.

## [References](#references)
[1] H. Lee and H. Kwon, “Contextual deep cnn based hyperspectral
classification,” in 2016 IEEE International
Geoscience and Remote Sensing Symposium (IGARSS),
2016, pp. 3322–3325.

[2] Y. Chen, H. Jiang, C. Li, X. Jia, and P. Ghamisi, “Deep
feature extraction and classification of hyperspectral images
based on convolutional neural networks,” IEEE
Transactions on Geoscience and Remote Sensing, vol.
54, no. 10, pp. 6232–6251, 2016.

[3] H. Sun, X. Zheng, X. Lu, and S. Wu, “Spectral–spatial
attention network for hyperspectral image classification,”
IEEE Transactions on Geoscience and Remote
Sensing, vol. 58, no. 5, pp. 3232–3245, 2020.

[4] S. Pu, Y. Wu, X. Sun, and X. Sun, “Hyperspectral image
classification with localized graph convolutional filtering,”
Remote Sensing, vol. 13, no. 3, 2021.

[5] D. Hong, L. Gao, J. Yao, B. Zhang, A. Plaza, and
J. Chanussot, “Graph convolutional networks for hyperspectral
image classification,” IEEE Transactions
on Geoscience and Remote Sensing, vol. 59, no. 7, pp.
5966–5978, 2021.


## [Datasets](#datasets)

- [Indian Pines Dataset](https://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes#Indian_Pines)
- [Pavia University Dataset](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes#Pavia_Centre_and_University)

## Run the repo

To run this project, follow these steps:

1. Clone the repository:

    ```shell
    git clone https://github.com/paulilioaica/ViT-GCN-RemoteSensing
    cd ViT-GCN-RemoteSensing
    ```

2. Install the required dependencies:

    ```shell
    pip install -r requirements.txt
    ```

3. Modify your configs.json params 

4. Populate your `data` folder with either datasets

4. Run 

    ```shell
    python main.py
    ```
