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

- - https://github.com/alexeygrigorev/clothing-dataset-small


## Model Selection

- For this project, several convolutional neural network architectures were evaluated for the clothing classification task. Since the dataset contains diverse clothing categories with variations in texture, shape, and appearance, choosing an appropriate pretrained model was an important step.

- Multiple pretrained models from the ImageNet family were tested and compared on the same dataset. These included architectures such as ResNet, MobileNet, and VGG. Each model was fine-tuned using transfer learning while keeping a similar training setup in order to ensure a fair comparison.

- After experimentation, Xception achieved the highest validation accuracy among the tested models. The architecture performed better at capturing fine-grained visual features present in clothing images.

- One of the reasons Xception works well for this task is its use of depthwise separable convolutions, which allow the network to learn spatial and channel-wise features more efficiently than standard convolution layers. This makes the model particularly effective for datasets where subtle visual differences between classes exist, such as distinguishing between shirts, t-shirts, and long sleeves.

- Because of its strong feature extraction capability and superior performance during experimentation, Xception was selected as the final backbone architecture for the classifier.

### Alternative Models to Consider

- Although Xception performed best in this project, several other architectures are commonly used for image classification and can also be effective depending on the dataset size and computational constraints:

- ResNet50 / ResNet101 – Strong baseline models with residual connections that help training deeper networks.

- MobileNetV2 / MobileNetV3 – Lightweight models suitable for deployment on mobile or low-resource environments.

- EfficientNet – Highly optimized architecture balancing accuracy and computational efficiency.

- VGG16 / VGG19 – Simpler architectures often used as baselines in transfer learning experiments.

- These architectures are widely recommended for transfer learning tasks and can serve as strong alternatives when experimenting with similar image classification problems.
- 
## Model Training Pipeline

- Load pretrained Xception base (include_top=False) for feature extraction.

- Apply GlobalAveragePooling2D to get vector representation.

- Add custom dense layers with dropout for regularization.

- Compile the model with Adam optimizer and Categorical Crossentropy.

- Train the model using data augmentation.

- Save checkpoints for the best performing model.

- Evaluate on validation data to monitor accuracy and overfitting.

## Results

Achieved high accuracy on the validation set without significant overfitting 88%.
- Test Accuracy = 91%
- Train Accuracy = 90%
