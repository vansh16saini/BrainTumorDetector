# Brain Tumor Detection Model 🧠

## About This Project
This is my **first deep learning model** created while learning about transfer learning and model training. The model uses **VGG16** as a base and has been fine-tuned to classify brain tumor images. This project helped me understand **data preprocessing, model training, and optimization techniques**.

## Features ✨
- **Transfer Learning:** Built on **VGG16**, a powerful pre-trained model.
- **Binary Classification:** Predicts whether a brain tumor is present or not.
- **Optimized Training:** Includes **dropout, batch normalization, and learning rate tuning** for better accuracy.

## Model Architecture 🏗
- **Base Model:** VGG16 (pretrained on ImageNet, top layer removed)
- **Custom Layers:**
  - Flatten Layer
  - Fully Connected Dense Layers
  - **Batch Normalization** for stable training
  - **Dropout** for overfitting prevention
  - Final **Sigmoid Layer** for binary classification
- **Optimizer:** AdamW with a tuned learning rate of `0.00005`
- **Loss Function:** Binary Crossentropy

## Training ⚡
- Trained on **203 images** (train set) and **50 images** (validation set)
- **Epochs:** 25
- **Batch Size:** 32
- **Validation Accuracy:** Improved through optimizations

## Results 
Training accuracy: ~74%
Validation accuracy: ~78%
Epochs: ~25

## Final Thoughts 💡
This project was a great learning experience in **transfer learning, model tuning, and deep learning basics**. If you're a beginner, feel free to 
**use, modify, or improve this model**!

It will significantly improve if I use more dataset as 203 is just a small number.

