"""
Self-made Linear regression model using only numpy
and importing no other libraries
"""

import numpy as np
from matplotlib import pyplot as plt
plt.style.use('dracula')


class LMS():
    """
    LMS (Least Mean Squares)
    """

    def __init__(self):
        """
        LMS fits a linear model with coefficients theta = (theta1, …, thetap) to
        minimize the residual sum of squares between the observed
        targets in the dataset, and the targets predicted by the linear approximation.
        """
        self.inputs = None
        self.targets = None
        self.theta = None
        self.losses = None

    @staticmethod
    def compute_hypothesis(inputs, theta):
        """
        Computes the hypothesis given an input and theta

        Parameters
        ----------
        x: matrix/array (preferably normalized)
        theta: vector of shape = number of features in input matrix
        """
        return inputs.dot(theta)

    @staticmethod
    def compute_loss(hypothesis, targets):
        """
        Computes the loss for linear regression (Least mean Squares)
        given a hypothesis and an actual output(target or y).

        Paramters
        ---------
        h: vector of size as target/y
        y: vector
        """
        return np.sum(np.square(targets-hypothesis))/2

    @staticmethod
    def normalize(inputs):
        """
        Normalizes the inputs

        Parameters
        ----------
        x: matrix/array
        """
        new_x = inputs.copy()
        mean = new_x.mean(axis=0)
        std = new_x.std(axis=0)
        new_x = (new_x-mean)/std
        new_x = np.insert(new_x, 0, 1, axis=1)
        return new_x

    def gradient_descent(self, alpha, epochs, init_theta):
        """
        Performs Gradient Descent on the given parameters and returns updated
        theta and a list of losses on each epoch

        Parameters
        ----------
        x: matrix/array
        y: vetor
        alpha: float
        epochs: int
        init_theta: vector
        """
        losses = []
        for _ in range(epochs):
            hypothesis = self.compute_hypothesis(self.inputs, init_theta)
            loss = self.compute_loss(hypothesis, self.targets)
            init_theta += alpha * ((self.targets-hypothesis).dot(self.inputs))
            losses.append(loss)
        return init_theta, losses

    def fit(self, inputs, targets, alpha=0.01, epochs=1500):
        """
        Normalizes the inputs by calling the 'normalize' function and then
        performs gradient descent by calling the 'gradient_descent' function

        Parameters
        ----------
        x: matrix/array
        y: vector
        alpha: float (default: 0.01)
        epochs: int (default: 1500)
        """
        try:
            init_theta = np.random.random(inputs.shape[1] + 1)
        except IndexError as shapes_not_right:
            raise ValueError("""Expected 2D matrix/array, got a 1D vector.
                                Reshape your data either using
                                array.reshape(-1, 1) if your data has a single
                                feature or array.reshape(1, -1) if it contains
                                a single sample""") from shapes_not_right
        self.inputs = self.normalize(inputs)
        self.targets = targets.copy()
        self.theta, self.losses = self.gradient_descent(
            alpha, epochs, init_theta)

    def predict(self, inputs):
        """
        Returns the prediction given a matrix/array x using theta parameters

        Parameters
        ----------
        x: matrix/array
        """
        inputs = self.normalize(inputs)
        return inputs.dot(self.theta)

    def evaluate_accuracy(self):
        """
        Evaluates accuracy by making predictions using the 'predict' function
        and then compares them to the actual targets

        Parameters
        ----------
        None
        """
        pred = self.inputs.dot(self.theta)
        return np.sum((1-abs(pred-self.targets)/self.targets)*100)/100

    def draw_loss(self):
        """
        Plots the losses against epochs
        """
        plt.plot(self.losses)
        plt.xlabel("epochs")
        plt.ylabel("Loss")
        plt.show()
