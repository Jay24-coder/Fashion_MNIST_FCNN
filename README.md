# Fashion_MNIST
### Problem Statement
Fashion MNIST is a dataset comprising 60,000 training images and 10,000 testing images, each depicting various fashion items in a 28x28 grayscale format. The dataset encompasses ten distinct categories of apparel. The primary aim of this assignment is to develop a fully connected neural network (FCNN) that can accurately classify these images into the correct fashion categories.

### Objectives:
#### Data Visualization and Preprocessing:
● Visualize the distribution of different fashion items in the dataset to
understand the data better.
● Normalize the image pixel values for effective model training.
● Convert the categorical labels into a one-hot encoded format to suit the
neural network’s output layer.
#### Model Architecture and Development:
● Design a fully connected neural network with appropriate input, hidden,
and output layers.
● Ensure the model is complex enough to capture the underlying patterns in
the data without overfitting.
#### Model Training:
● Train the model on the preprocessed training data, using categorical
cross-entropy as the loss function, stochastic gradient descent as the
optimizer, and accuracy as the evaluation metric.
● Utilize a validation split to monitor the model’s performance on unseen
data during training.
#### Performance Evaluation:
● Assess the model’s performance using both training and validation
datasets.
● Visualize the training and validation loss and accuracy to identify any signs
of overfitting or underfitting.
