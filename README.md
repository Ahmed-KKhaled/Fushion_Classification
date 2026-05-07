# 👕 Fashion Classification with Xception

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-FF6F00?logo=tensorflow&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-0.12+-teal)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
## 📌 Overview
This project focuses on **multi-class clothing image classification** using **Transfer Learning** with the **Xception architecture**.  
The model classifies images into **10 clothing categories**:
- Dress  
- Hat  
- Long Sleeve  
- Outwear  
- Pants  
- Shirt  
- Shoes  
- Shorts  
- Skirt  
- T-shirt  
---
## 🧠 Model Architecture
The model is built on top of a **pretrained Xception network (ImageNet weights)** with custom modifications:
- Removed the top classification layers (`include_top=False`)
- Added:
  - Global Average Pooling layer
  - Fully connected (Dense) layers
  - Dropout for regularization
- Final layer uses **Softmax activation** for multi-class classification
---
## ⚙️ Key Techniques
### 🔹 Transfer Learning
- Leveraged pretrained Xception as a feature extractor
- Fine-tuned upper layers to adapt to the clothing dataset
### 🔹 Data Augmentation
Applied to improve generalization:
- Rotation
- Horizontal flipping
- Zooming
- Shifting
### 🔹 Regularization
- Dropout layers added to reduce overfitting
### 🔹 Learning Rate Tuning
- Optimized using **Adam optimizer**
- Learning rate adjustments improved convergence stability
### 🔹 Checkpointing
- Saved best model weights during training
- Prevented loss of progress and enabled best model selection
### 🔹 Global Average Pooling
- Reduced model parameters
- Improved generalization over fully connected flattening
---
## 📂 Dataset
- Contains labeled clothing images across 10 categories
- Images were resized and preprocessed to match Xception input format  
🔗 Dataset:  
[Data](https://github.com/alexeygrigorev/clothing-dataset-small)
---
## 🏗️ Training Pipeline
1. Load pretrained Xception model (`include_top=False`)
2. Extract features using convolutional base
3. Apply GlobalAveragePooling2D
4. Add custom dense + dropout layers
5. Compile model:
   - Optimizer: Adam  
   - Loss: Categorical Crossentropy  
6. Train with data augmentation
7. Save best model using checkpoints
8. Evaluate on validation set
---
## 📊 Results

### Xception (Final Model)
- ✅ **Train Accuracy:** 90%  
- ✅ **Validation Accuracy:** 88%  
- ✅ **Test Accuracy:** 91%  

### VGG16
- 📌 **Train Accuracy:** 80%  
- 📌 **Validation Accuracy:** 84%  
- 📌 **Test Accuracy:** 87%  

### ResNet50
- 📌 **Train Accuracy:** 90%  
- 📌 **Validation Accuracy:** 87%  
- 📌 **Test Accuracy:** 90%  

---

## 🔍 Model Comparison

| Model       | Train Acc | Val Acc | Test Acc |
|-------------|:---------:|:-------:|:--------:|
| VGG16       |    80%    |   84%   |   87%    |
| ResNet50    |    90%    |   87%   |   90%    |
| **Xception**|  **90%**  | **88%** | **91%**  |

> 🏆 **Xception** achieved the best overall performance across all metrics.

---

## 🔍 Why Xception?
Xception outperformed other models due to:
- Depthwise separable convolutions  
- Better spatial + channel feature extraction  
- Strong performance on fine-grained differences (e.g. shirt vs t-shirt)
---
## 🚀 Future Improvements
- Hyperparameter tuning (batch size, learning rate schedules)
- Fine-tuning deeper layers of Xception
- Using larger dataset for better generalization
- Experimenting with EfficientNet / Vision Transformers
- Deploying model using Flask or FastAPI
---
## 🛠️ Tech Stack
- Python  
- TensorFlow / Keras  
- NumPy / Pandas  
- Matplotlib  
---
## 📬 Contact
If you have any questions or suggestions, feel free to reach out!
email : ahmedkhaled5.ml@gmail.com
