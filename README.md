# Handwritten Character Recognition

This project focuses on recognizing handwritten characters using Convolutional Neural Networks (CNN). It involves data preprocessing, model training, and evaluation, leveraging various Python libraries for efficient implementation.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Results](#results)

## Introduction

Handwritten Character Recognition (HCR) is a crucial application of machine learning in the field of Optical Character Recognition (OCR). It is used in various real-world scenarios such as postal mail sorting, bank check processing, and form data entry. This project demonstrates a robust approach to recognizing handwritten characters using CNNs, providing a foundation for further advancements and applications.

## Features

- **Data Preprocessing**: Includes normalization, reshaping, and one-hot encoding of the dataset.
- **Model Training**: Utilizes Convolutional Neural Networks to learn and classify handwritten characters.
- **Model Evaluation**: Evaluates the performance of the model using accuracy, confusion matrix, and visualizations.

## Installation

### Prerequisites

Ensure you have Python 3.6+ installed on your system. Additionally, you will need the following libraries:

- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- tensorflow
- keras

### Installation Steps

1. Clone the repository:

    ```bash
    git clone https://github.com/harshit64830/Handwritten-Character-Recognition.git
    cd Handwritten-Character-Recognition
    ```

2. Install the required libraries:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Running the Notebook

1. Open the Jupyter Notebook:

    ```bash
    jupyter notebook Handwritten_Character_Recognisation.ipynb
    ```

2. Execute the cells in the notebook sequentially to preprocess the data, train the model, and evaluate the results.

## Model Architecture

The model is constructed using Convolutional Neural Networks (CNN), known for their prowess in image recognition tasks. The architecture typically includes:

- **Convolutional Layers**: To extract features from the input images.
- **Pooling Layers**: To down-sample the spatial dimensions.
- **Fully Connected Layers**: For classification purposes.
- **Activation Functions**: Such as ReLU and Softmax for non-linearity and probability distribution.

### Detailed Architecture

1. **Input Layer**: Takes the image input.
2. **Convolutional Layer 1**: Applies multiple filters to capture different features.
3. **Max-Pooling Layer 1**: Reduces dimensionality while retaining important features.
4. **Convolutional Layer 2**: Further feature extraction.
5. **Max-Pooling Layer 2**: Further reduction of dimensionality.
6. **Flattening Layer**: Converts the 2D matrix into a vector.
7. **Fully Connected Layer**: Dense layer for classification.
8. **Output Layer**: Produces probability distribution over classes.

## Dataset

The dataset used is a collection of grayscale images of handwritten characters. Each image is labeled with the corresponding character. The dataset is split into training and testing sets to evaluate the model's performance.

## Results

The model achieves high accuracy of above 95% in recognizing handwritten characters. The results are visualized in the notebook, showing the performance metrics and sample predictions.
