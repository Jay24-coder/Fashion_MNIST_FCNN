# 🧥 Fashion MNIST Classification with Keras

This project demonstrates a multi-layer neural network built using Keras to classify images from the Fashion MNIST dataset. The dataset consists of grayscale 28x28 images of clothing items, and the model predicts which category an item belongs to out of 10 predefined classes.

---

## 📂 Project Structure

- Loads and preprocesses the Fashion MNIST dataset
- Normalizes and reshapes image data for training
- One-hot encodes labels
- Builds and trains a deep fully-connected neural network
- Evaluates performance with accuracy and loss plots
- Makes predictions on sample training data

---

## 📊 Dataset

**Fashion MNIST** dataset is a replacement for the classic MNIST digits dataset, consisting of:

- 60,000 training images  
- 10,000 test images  
- 10 clothing categories:

  1. T-shirt/top  
  2. Trouser  
  3. Pullover  
  4. Dress  
  5. Coat  
  6. Sandal  
  7. Shirt  
  8. Sneaker  
  9. Bag  
  10. Ankle boot  

---

## 🧪 Model Architecture

- **Input Layer**: 784 neurons (flattened 28x28 image)  
- **Hidden Layers**: Five Dense layers with 80 units and ReLU activation  
- **Dropout**: 8% applied after each hidden layer to reduce overfitting  
- **Output Layer**: 10 neurons with softmax activation (for multi-class classification)  

---

## 🧠 Training

- **Optimizer**: Stochastic Gradient Descent (SGD)  
- **Loss Function**: Categorical Crossentropy  
- **Batch Size**: 25  
- **Epochs**: 50  
- **Validation**: Uses test set for validation metrics  

---

## 📈 Results

Training and validation accuracy/loss are plotted to monitor performance over 50 epochs. Example visualizations include:

- Training vs. Validation Loss  
- Training vs. Validation Accuracy  

These plots help assess overfitting and generalization.

---

## 🔍 Prediction

Sample prediction on the training set:

```python
print(pred[88])             # Shows softmax probabilities
print(np.argmax(pred[88])) # Predicted class index
```

--- 🛠️ Requirements
Make sure you have the following libraries installed:
```
pip install numpy pandas matplotlib seaborn tensorflow

```

---

## 📌 Notes
- This model is kept intentionally simple for educational purposes.
- You are encouraged to experiment with other architectures (e.g., CNNs), optimizers, or regularization methods to boost performance.

