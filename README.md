# MNIST

# Problem Statement:

This assignment has two problems.

P1. Implement a simple 1 or 2 hidden layer MLP USING any deep learning library for predicting MNIST images.

P2. Implement a simple CNN USING any deep learning library for predicting MNIST images.

Instructions:
1. Download the MNIST dataset from http://yann.lecun.com/exdb/mnist/ (four files).
2. Extract all the files into a folder named 'data' just outside the folder containing the main.py file. This code reads the data files from the folder '../data'.
3. The functions for training the model is implemented in the train_dense.py using DNN and train_cnn.py using CNN. You might also create other functions for your convenience, but do not change anything in the main.py file or the function signatures of the train and test functions in the train files.
4. The train function must train the neural network given the training examples and save them in a folder named 'weights' in the same folder as main.py.
5. The test function read the saved weights and given the test examples it must return the predicted labels.
