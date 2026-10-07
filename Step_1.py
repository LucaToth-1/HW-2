# Luca Toth and Brayden Molinyawe put in both equal effort on this assingment.

import numpy as np
import matplotlib.pyplot as plt 

from iris_utils import *
X, y = load_iris_csv("iris.csv")
Xtr_raw, Xte_raw, ytr, yte = stratified_split(X, y, test_fraction=0.3, seed=0)
Xtr, Xte, mu, sd = standardize(Xtr_raw, Xte_raw)


# Part 1: Binary Logistic Regression

# Sigmoid Function
# will squash any real number "z" to a value between 0 and 1.
# It works on a single number or a numpy array of numbers.
def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

# binary logistic loss function
# Computes the binary logistic loss (cross-entropy loss) with L2 regularization.
# A single number measuring how bad the current weights are
# it has 2 parts:
#   cross-entropy loss: -1/n * sum(y * log(p) + (1-y) * log(1-p))
#   L2 regularization term: 0.5 * lam * sum(theta[1:] ** 2) (we don't regularize the bias term theta[0])
def binary_logistic_loss(theta, Xb, y, lam):
    p = np.clip(sigmoid(Xb @ theta), 1e-12, 1 - 1e-12) # we never want to log 0
    ce = -np.mean(y * np.log(p) + (1 -y) * np.log(1 - p)) # cross entropy loss
    return ce + 0.5 * lam * np.sum(theta[1:] ** 2) # Add regularization term, but not for bias (unpenalized)


# Binary logistic gradient function
# computes the gradient of the binary logistic loss function with L2 regularization.
# Basically, we are computing the way the loss from binary logistic loss
# moves the fastest, or the biggest  negative derivative. This is used to update the weights in the direction that reduces the loss.
#The First term is the average prediction error for each feature, 
#  and the second term is the regularization term that penalizes large weights (except for the bias term).
def binary_gradient(theta, Xb, y, lam):
    n = Xb.shape[0]
    p = sigmoid(Xb @ theta)
    theta_tilde = theta.copy()
    theta_tilde[0] = 0 # don't regularize the bias term
    return Xb.T @ (p - y) / n + lam * theta_tilde

# Binary logistic regression training class
#This is a complete binary classifier, so we are labeling the data into 1's and 0's.
# trained with batch gradint descent.
class BinaryLogisticRegression:
    def __init__(self, lr=0.5, lam=0.1, max_iter=2000):
        self.lr, self.lam, self.max_iter = lr, lam, max_iter
        self.theta = None
        self.loss_history = []

    # fit function
    # adds a bias column, starts the weight at 0, then for max_iter,
    # it computes and records the current loss, and updates the weights using the gradient of the loss function.
    def fit(self, X, y): 
        Xb = add_bias(X)
        self.theta = np.zeros(Xb.shape[1]) # initialize weights to zero
        self.loss_history = []
        for i in range(self.max_iter):
            self.loss_history.append(binary_logistic_loss(self.theta, Xb, y, self.lam))
            self.theta -= self.lr * binary_gradient(self.theta, Xb, y, self.lam) # update weights using gradient descent
        return self

    # predict_proba function
    # Returns P(label=1) for each sample in X
    def predict_proba(self, X):
        return sigmoid(add_bias(X) @ self.theta)

    # predict function
    # returns 1 if the probility is greater than or equal to 0.5, else it returns 0.
    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

# Part 2.1 : two binary classifiers for the 3 class iris dataset. Chained with the chain rule

# TwoStageClassifer class
#this class solves the 3 class problem with 2 binary models.
# Stage 1: answers if the species is Setosa or not. If it is Setosa, we are done. If not, we go to stage 2.
# stage 2: answers if the species is Versicolor or virginica.
#  It never sees Setosa, so it only needs to distinguish between Versicolor and Virginica.
class TwoStageClassifier:
    def __init__(self, lr=0.5, lam=0.1, max_iter=2000):
        self.stage1 = BinaryLogisticRegression(lr, lam, max_iter) #again, this is seteso vs rest of the species
        self.stage2 = BinaryLogisticRegression(lr, lam, max_iter)  # and this is versicolor vs virginica

    #fit function
    #trains both stages. Here are the labels for each stage:
    # stage 1: 1 if Setosa
    # stage 2: 1 if Versicolor, 0 if Virginica
    def fit(self, X, y):
        #Stage 1: Setosa vs not Setosa
        self.stage1.fit(X, (y == 0).astype(int))
        # stage 2: Versicolor vs Virginica (only train on non-Setosa samples)
        mask = (y != 0)
        self.stage2.fit(X[mask], (y[mask] == 2).astype(int))
        return self

    # predict_proba function
    # we are now combining the stages with the chain rule.
    # P(setosa) = q1
    # P(versicolor) = (1-q1) * (1 - q2)
    # P(virginica) = (1-q1) * q2
    # where q1 = P(setosa) and q2 = P(virginica | not setosa) from stage 2.
    #each row sums up to 1
    def predict_proba(self, X):
        q1 = self.stage1.predict_proba(X) # P(setosa | X)
        q2 = self.stage2.predict_proba(X) # P(virginica | X, not setosa)
        return np.column_stack((q1, (1 - q1) * (1 - q2), (1 - q1) * q2)) 

    # Predict function
    # This function just picks the class with the highest probability
    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


# Part 2.2: Softmax Regression

# Softmax function
# this function turns each row of raw class scores (S) into 3 probabilities that sum to 1 and are positive.
def Softmax(S):
    S = S - S.max(axis=1, keepdims=True)
    E = np.exp(S)
    return E / E.sum(axis=1, keepdims=True)

# softmax loss function
# This is the 3 class version of the binary logistic loss function.
# It computes the cross-entropy loss for multi-class classification with L2 regularization.
#the first term is the average cross-entropy loss, and the second term is the L2 regularization term.
def softmax_loss(W, Xb, y, lam):
    P = np.clip(Softmax(Xb @ W), 1e-12, 1)
    ce = -np.mean(np.sum(Y * np.log(P), axis=1))
    return ce + 0.5 * lam * np.sum(W[1:] ** 2) # Add regularization term, but not for bias 


# softmax gradient function
# This function computes the gradient of the softmax loss function with L2 regularization.
# it has the same structure as the binary gradient function, but it works for multiple classes.
def softmax_gradient(W, Xb, y, lam):
    n = Xb.shape[0]
    P = Softmax(Xb @ W)
    W_tilde = W.copy()
    W_tilde[0] = 0 # don't regularize the bias term
    return Xb.T @ (P - Y) / n + lam * W_tilde


# Softmax Regression class
# this class implements the softmax regression model for multi-class classification.
# EXACTLY like BinaryLogisticRegression, but with a weight matrix W instead of a weight vector theta, 
# and using the softmax loss and gradient functions.
class SoftmaxRegression:
    def __init__(self, lr=0.5, lam=0.1, max_iter=3000, k =3 ):
        self.lr, self.lam, self.max_iter, self.k = lr, lam, max_iter, k
        self.W = None
        self.loss_history = []

    # Fit function
    # one-hot encodesn the labels, starts W at zero, then repeates max_iter times:
    # we then record the loss, update W -=lr * gradient, and return self.
    def fit(self, X, y):
        Xb = add_bias(X)
        Y = one_hot(y, self.k)
        self.W = np.zeros((Xb.shape[1], self.k)) # initialize weights
        self.loss_history = []
        for i in range(self.max_iter):
            self.loss_history.append(softmax_loss(self.W, Xb, Y, self.lam))
            self.W -= self.lr * softmax_gradient(self.W, Xb, Y, self.lam) # update weights using gradient descent
        return self

    # predict_proba function
    # this returns an n x 3 array of class probabilities for each sample in X.
    def predict_proba(self, X):
        return Softmax(add_bias(X) @ self.W)

    # predict function
    # this returns the class with the highest probability for each sample in X.
    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)