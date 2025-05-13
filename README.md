# 🧠 Keras MNIST MLP

This project implements a simple **Multi-Layer Perceptron (MLP)** neural network using **TensorFlow/Keras** to classify handwritten digits (0–9) from the **MNIST** dataset.

It includes data preprocessing, model training, evaluation, and accuracy visualization — ideal for beginners in deep learning and computer vision.

---

## 🎯 Purpose of the Model

This machine learning model is built for **image classification** — specifically, to recognize **handwritten digits** from images. It serves as an introduction to neural networks and the foundations of **supervised learning**.

### Use Cases:
- Digit recognition in forms and postal codes
- Teaching basic neural network design
- Benchmarking and comparing model performance
- Preparing for more advanced deep learning tasks like CNNs

---

## ⚙️ Why TensorFlow and Keras?

- **TensorFlow** is an open-source machine learning framework developed by Google.
- **Keras** is a high-level API within TensorFlow that simplifies the process of building and training deep learning models.

Keras is ideal for:
- Rapid prototyping
- Educational use
- Building models with clean, readable code

> In this project, Keras handles all neural network operations — from defining the architecture to training and evaluation.

---

## 📂 Dataset Overview – MNIST

- **Name**: Modified National Institute of Standards and Technology (MNIST)
- **Size**: 70,000 grayscale images (60,000 for training, 10,000 for testing)
- **Image Size**: 28x28 pixels
- **Classes**: 10 (digits 0 through 9)
- **Type**: Supervised learning / classification

---

## 🚀 How It Works

1. **Load and normalize data**  
   - Scales pixel values from 0–255 to 0–1

2. **Model architecture**  
   - `Flatten`: converts 28x28 images into 784-length vectors  
   - `Dense(128, relu)` + `Dropout(0.2)`: hidden layer with dropout to prevent overfitting  
   - `Dense(10, softmax)`: output layer for 10-class classification (0–9)

3. **Training**  
   - Optimizer: `Adam` (adaptive learning rate)  
   - Loss function: `sparse_categorical_crossentropy`  
   - Metric: `accuracy`  
   - 5 training epochs, 10% validation split

4. **Evaluation**  
   - Assessed on unseen test set  
   - Outputs overall accuracy score

5. **Visualization**  
   - Accuracy over epochs for training and validation

---

## 📊 Sample Output

```text
Formato de x_train: (60000, 28, 28)
Formato de y_train: (60000,)
Acurácia no teste: 0.9785
````

![Training Accuracy Plot](https://user-images.githubusercontent.com/your-image-link-here.png) <!-- Optional: insert your own plot -->

---

## 📦 Requirements

Install the required packages:

```bash
pip install tensorflow matplotlib
```

Or use this `requirements.txt`:

```txt
tensorflow==2.16.1
matplotlib==3.8.4
```

---

## 📄 License

This project is licensed under the MIT License.

---

## 👤 Author

Developed by [Peterson Chiquetto](https://github.com/petersonchiquetto)

---

## 🔗 Run in Google Colab

[📓 Click here to open in Colab](https://colab.research.google.com/drive/1J_IshgKb1R40BQGk68HAowDkfvmKq8xH)

```
