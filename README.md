# **Filtering the Slop: Detecting AI Images and their Features**

Jiarui Chen
Murali Krishtna J
Navarro Jose

## Table of Contents

1. [Overview](#Overview)
2. [Installation](#Installtion)
3. [Running](#Running)
4. [Acknowledgement](#Acknowledgement)

## Overview

​	Training a classifier model to encode and classify between real images and AI generated images does not work since these models will latch onto the features of a specific generative family of models and detect everything else as real. Utilizing a pretrained network instead, such as CLIP:VIT, as an encoder for the image and classifying on that feature space that was specifically not trained on the classification task provides the best results and generalizes across different models families. So far, KNN and Linear Probing methods have been employed with good results, implying that in this highly dimensional feature space, there is a degree of linear separability between the real and fake encoded features. In this case, a more robust classifier model or encoding architecture might provide better results, such as a Soft Margin Support Vector machine, or a DINO encoder that has shown benefits in KNN classification in the past.

## Installation

1. Clone the repository

    First, clone the repository: 

2. Required Environment

    The project requires the following Python libraries to run:

    - torch
    - torchvision
    - PIL
    - scikit-learn
    - cv2

    Install them using this command: 

    ```bash
    pip install torch torchvision scikit-learn opencv-python
    ```

    

3. Dataset

    A diffusion dataset with 10 categories, with 1,000 images per category is used. The dataset can be downloaded at [here](https://drive.google.com/file/d/1FXlGIRh_Ud3cScMgSVDbEWmPDmjcrm1t/view). 

    After download, unzip it to a folder named `datasets`. The path will be:
    ```
    datasets
    └── diffusion_datasets			
           ├── dalle
           │── glide_50_27
           │── glide_100_10
           │──    .
           │──    .
    ```

    

4. 

## How to run

You can run the repo for a single image test - 

bash run.sh SINGLE_IMAGE

You can also provide the base model/classifier like -
- bash run.sh SINGLE_IMAGE DINO:vit_b_16 SVM
- bash run.sh SINGLE_IMAGE CLIP:ViT-L/14 Linear

For SINGLE_IMAGE, you'll find the results in the singe_image_results folder. Please note that, as mentioned in the report, the GRAD CAM highlight does not *always* produce human-intuitive heatmaps

## Acknowledgement

​	This project is based on [*Detecting fake images*](https://utkarshojha.github.io/universal-fake-detection/) project of *Utkarsh Ojha*, *Yuheng Li* and *Yong Jae Lee*. 