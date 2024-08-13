# Deep Learning Projects

This repository contains two advanced deep learning projects, each designed to tackle different multi-class classification tasks using various machine learning and deep learning techniques. Both projects involve practical applications, detailed data visualization, and model implementation from scratch, as well as using deep learning frameworks like PyTorch.

## Project 1: STL-10 Image Classification- Classification task with different types of NN using PyTorch

### Overview
This project involves solving a multi-class classification task on the STL-10 dataset using various types of neural networks implemented in PyTorch. The STL-10 dataset consists of 96x96 pixel images from 10 different classes: Airplane, Bird, Car, Cat, Deer, Dog, Horse, Monkey, Ship, and Truck.

### Key Tasks
1. **Data Visualization:** Display 4 different examples from each class in a 10x4 grid.
2. **Network Implementations:** Implement and train the following models:
   - Logistic regression over flattened images.
   - Fully-connected Neural Network (NN) with at least 3 hidden layers.
   - Convolutional Neural Network (CNN) with at least two convolutional layers.
   - Pre-trained MobileNetV2 as a feature extractor with fixed weights.
   - Pre-trained MobileNetV2 with learned weights.
3. **Data Augmentation:** Apply various data augmentation techniques to improve model performance.
4. **Hyperparameter Tuning:** Experiment with different configurations of hyperparameters to optimize model performance.
5. **Reporting:** Submit a detailed report including data visualizations, loss and accuracy curves, and key insights from the experiments.

### Files
- `run_mab_advertisement.py`: Implements and runs the Advertisement Optimization Bandit.
- `run_mab_personalization.py`: Implements and runs the Content Personalization Bandit.
- `report.pdf`: A detailed report documenting the data visualization, model architectures, training process, and results.

### Dataset
- [STL-10 Dataset](https://cs.stanford.edu/~acoates/stl10/)

---

## Project 2: Fashion-MNIST Classification-Classification task with a Logistic Regression Classifier and a NN

### Overview
This project involves implementing a multi-class classification task on the Fashion-MNIST dataset using a logistic regression classifier and a Neural Network (NN) with one hidden layer. Both models are implemented from scratch using only Numpy, without relying on deep learning frameworks like PyTorch or TensorFlow.

### Key Tasks
1. **Data Visualization:** Display 4 different examples from each class in a 10x4 grid.
2. **Logistic Regression Classifier:** Implement and train a multi-class logistic regression model, including softmax activation, L2 regularization, and mini-batch gradient descent.
3. **Neural Network Implementation:** Implement a Neural Network with one hidden layer and derive the gradients for the model parameters.
4. **Hyperparameter Tuning:** Experiment with different configurations of batch size, learning rate, and regularization coefficients to optimize model performance.
5. **Reporting:** Submit a detailed report including data visualizations, loss and accuracy curves, and key insights from the experiments.

### Files
- `train.csv`: Training data for the Fashion-MNIST dataset.
- `test.csv`: Test data for the Fashion-MNIST dataset.
- `lr_pred.csv`: Predictions from the logistic regression model on the test dataset.
- `report.pdf`: A detailed report documenting the data visualization, model architectures, training process, and results.

### Dataset
- [Fashion-MNIST Dataset](https://github.com/zalandoresearch/fashion-mnist)

---

## How to Use

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/your-username/deep-learning-projects.git
    ```

2. **Navigate to the Project Directory:**
    ```bash
    cd deep-learning-projects
    ```

3. **Set Up the Environment:**
   For the STL-10 project, make sure you have Python and the required libraries installed. Use the following command to install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. **Run the Projects:**
   - For the STL-10 project, run the corresponding Python scripts for each model.
   - For the Fashion-MNIST project, use the provided Numpy implementations to train and evaluate the models.

## Contact

For any questions or feedback, please contact [shoahmgalili1@gmail.com](mailto:shoahmgalili1@gmail.com).
