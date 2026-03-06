# Fushion_Classification 👕

## Clothing Classification using Xception and Transfer Learning

This project focuses on classifying clothing images into 10 categories:

- dress

- hat

- longsleeve

- outwear

- pants

- shirt

- shoes

- shorts

- skirt

- tshirt

## The classification model is based on Xception pretrained on ImageNet, with modifications for the clothing dataset.

### Key Techniques Used

- During the development of this project, several important techniques were applied to improve model performance and generalization:

- Transfer Learning

- The convolutional base of Xception was used to extract features.

- Fully connected layers were replaced to adapt to the clothing dataset.

- Data Augmentation

- Techniques like rotation, horizontal flips, zooming, and shifting were applied to the training dataset.

- This helps the model generalize better and reduces overfitting.

- Adjusting the Learning Rate

- The Adam optimizer with a tuned learning rate was used.

- Learning rate adjustments were applied to stabilize training and improve convergence.

- Dropout Regularization

- Dropout layers were added between dense layers to prevent overfitting.

- Checkpointing

- Model checkpoints were used to save the best model weights during training.

- Ensures recovery in case of interruptions and allows selecting the best performing model.

- Global Average Pooling

- Applied to the convolutional feature maps before the dense layers.

- Reduces the number of parameters and summarizes features effectively.

- Categorical Crossentropy with Logits

- Used for multi-class classification.


## Dataset

- The dataset contains labeled images of clothing items across 10 categories.

- Images were resized and preprocessed to match the input requirements of Xception.

## Model Training Pipeline

- Load pretrained Xception base (include_top=False) for feature extraction.

- Apply GlobalAveragePooling2D to get vector representation.

- Add custom dense layers with dropout for regularization.

- Compile the model with Adam optimizer and Categorical Crossentropy.

- Train the model using data augmentation.

- Save checkpoints for the best performing model.

- Evaluate on validation data to monitor accuracy and overfitting.

## Results

Achieved high accuracy on the validation set without significant overfitting 88.
- Test Accuracy = 91
- Train Accuracy = 90

Requirements

Python 3.10+

tensorflow>=2.12.0,<2.21.0

numpy

pillow

matplotlib
