
import numpy as np

class lms(object):
    def __init__(self):
        self.theta = np.random.random(2)
        pass

    def computeCost(self, x):
        return x.dot(self.theta)

    def fit(self, x, y, alpha, epochs=1500):
        x = x.reshape(len(x), 1)
        x = np.insert(x, 0, 1, axis=1)
        for i in range(epochs):
            h = self.computeCost(x)
            self.theta += alpha * (y - h).dot(x)
        pass

    def predict(self, x):
        x = x.reshape(len(x), 1)
        x = np.insert(x, 0, 1, axis=1)
        return x.dot(self.theta)

