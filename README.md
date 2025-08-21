# MNIST Digit Classifier — Deep Neural Networks  

**Author:** Miltiadis Lefkopoulos  

---

## 📌 Project Description  
This project focuses on the **classification of handwritten digits (0–9)** using the **MNIST dataset**, which contains 60,000 grayscale images (28×28×1 pixels).  

The main objective is to train and evaluate **Deep Neural Networks (DNNs)** built with **TensorFlow**, performing hyperparameter tuning, experimenting with network architectures, and applying workflow improvements.  

---

## 🚀 Workflow  

1. **Data Loading & Visualization**  
   - Loaded the MNIST dataset from TensorFlow.  
   - Displayed random digits with their corresponding labels.  

2. **Preprocessing**  
   - Flattened 28×28 images into 784-dimensional vectors.  
   - One-hot encoded labels for 10 classes.  
   - Normalized pixel values to range [0,1] for improved training.  

3. **Baseline Model**  
   - Input: 784 features  
   - Two hidden layers with 256 units each (tanh activation)  
   - Output: 10 units with softmax activation  
   - Optimizer: SGD (learning rate = 0.001)  
   - Epochs: 10  

4. **Hyperparameter Tuning**  
   - Tested different architectures and parameters:  
     - Example 1: relu, lr=0.001, 20 epochs, [128, 64] hidden layers  
     - Example 2: tanh, lr=0.01, 15 epochs, [64, 64, 32] hidden layers  

5. **Improvements**  
   - Normalization of input data (0–1).  
   - Added **Batch Normalization** to stabilize and accelerate training.  
   - Applied **Dropout (0.2)** to reduce overfitting.  

6. **Evaluation**  
   - Trained models were evaluated on the MNIST test set.  
   - Visualized **loss and accuracy per epoch**.  
   - Displayed one **misclassified digit per class**.  

---

## 🏗️ Final Model Architecture  
- Input: 784 features  
- Hidden Layer 1 → Dense (256 units) → BatchNorm → tanh → Dropout(0.2)  
- Hidden Layer 2 → Dense (256 units) → BatchNorm → tanh → Dropout(0.2)  
- Output Layer → Dense (10 units, softmax)  

Optimizer: **SGD** (lr=0.001)  
Loss: **Categorical Crossentropy**  
Epochs: **10**  

---

# Documentation

You can [view the full documentation as a PDF](./report.pdf).