# Dogs vs. Cats Classification using HOG and SVM

This repository contains the code and a detailed explanation of a machine learning project to classify images of dogs and cats. The approach utilizes the Histogram of Oriented Gradients (HOG) for feature extraction and a Linear Support Vector Machine (SVM) for the classification task.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
  - [1. Feature Extraction: HOG](#1-feature-extraction-hog)
  - [2. Model Training: SVM](#2-model-training-svm)
- [Performance](#performance)
- [How to Use](#how-to-use)
- [Dependencies](#dependencies)

---

## Project Overview

The primary goal of this project is to build a classifier that can accurately distinguish between images of dogs and cats. This is a classic computer vision problem that serves as an excellent introduction to image classification techniques. Instead of using deep learning, this project employs a traditional machine learning pipeline.

---

## Dataset

The project uses the "Dogs vs. Cats" dataset from a Kaggle competition. This dataset contains thousands of images of dogs and cats, which are used for training and testing the model.

- **Training Set**: Contains 25,000 images, equally split between dogs and cats. A subset of this data is used for training the model.
- **Test Set**: Contains 12,500 images, which are used to evaluate the model's performance and generate a submission file.

The dataset is downloaded and unzipped using the Kaggle API within the provided Jupyter Notebook.

---

## Methodology

The core of this project is a two-step process: extracting meaningful features from the images and then training a classifier on these features.

### 1. Feature Extraction: HOG

**Histogram of Oriented Gradients (HOG)** is a feature descriptor used in computer vision for object detection. It works by counting occurrences of gradient orientation in localized portions of an image. This method is particularly effective for capturing shape and texture.

For each image, the following steps are performed:
1.  The image is resized to a standard size of 64x128 pixels.
2.  It is then converted to grayscale.
3.  HOG features are extracted, capturing the essential visual information.

### 2. Model Training: SVM

A **Linear Support Vector Machine (SVM)** is used as the classifier. SVMs are powerful and versatile supervised machine learning models, capable of performing linear or non-linear classification. In this project, a `LinearSVC` is employed, which is a fast and efficient implementation of a linear SVM.

The training process involves:
1.  **Splitting the Data**: The extracted features and their corresponding labels (0 for cat, 1 for dog) are split into training and testing sets.
2.  **Creating a Pipeline**: A `Pipeline` from scikit-learn is used to chain the `StandardScaler` (for feature scaling) and the `LinearSVC` classifier.
3.  **Training**: The pipeline is trained on the training data using the `.fit()` method.

---

## Performance

The model's performance is evaluated on the test set created from the initial data split. The key metrics are:

-   **Accuracy**: The model achieved an accuracy of approximately **64.50%** on the validation set.

-   **Classification Report**:

|           | precision | recall | f1-score | support |
| :-------- | :-------: | :----: | :------: | :-----: |
| **Cat** |   0.66    |  0.61  |   0.63   |   200   |
| **Dog** |   0.64    |  0.68  |   0.66   |   200   |
| **macro** |   0.65    |  0.65  |   0.64   |   400   |
| **weighted**|   0.65    |  0.65  |   0.64   |   400   |


---

## How to Use

To replicate this project, you can follow these steps:

1.  **Clone the Repository**:
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```

2.  **Upload to Google Colab**: Upload the `Dogs_vs_Cats.ipynb` notebook to Google Colab.

3.  **Kaggle API Key**:
    -   Go to your Kaggle account, and under the "API" section, click "Create New API Token". This will download a `kaggle.json` file.
    -   When you run the first cell of the notebook, you will be prompted to upload this `kaggle.json` file.

4.  **Run the Notebook**: Execute the cells in the notebook in order. The notebook will handle:
    -   Setting up the Kaggle API.
    -   Downloading and unzipping the dataset.
    -   Preprocessing the images and extracting HOG features.
    -   Training the SVM model.
    -   Evaluating the model.
    -   Generating a `submission.csv` file with predictions for the test set.

---

## Dependencies

The project is designed to be run in a Google Colab environment, which comes with most of the necessary libraries pre-installed. The key Python libraries used are:

-   `numpy`
-   `pandas`
-   `opencv-python` (`cv2`)
-   `scikit-image`
-   `scikit-learn`
-   `matplotlib`
-   `tqdm`

These can be installed via pip if you are running the project locally:
```bash
pip install numpy pandas opencv-python scikit-image scikit-learn matplotlib tqdm
```
 ## Developed By

* **Aditya Dasappanavar**
* **GitHub:** [AdityaD28](https://github.com/AdityaD28)
* **LinkedIn:** [adityadasappanavar](https://www.linkedin.com/in/adityadasappanavar/)
