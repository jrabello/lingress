import pandas as pd
import matplotlib.pyplot as plt
from gradient_decent import *


class LinearRegression:
    """        
        data = two features(x,y) each n-dimentional -> ex:[[1,2],[3,4]]
        m = initial slope
        b = initial y-intercept        
        y = b + m*x
    """

    def __init__(self, points, b = 0, m = 0, iterations = 100, learning_rate = 0.01):
        self.b = b        
        self.m = m
        self.points = points
        self.iterations = iterations
        self.gd = GradientDescent(learning_rate)


    def compute(self):
        for i in range(0, self.iterations):
            self.b, self.m = self.gd.step(self.points, self.b, self.m)
            #print("{} {} {}".format(self.sse(), self.w0, self.w1))


    def sse(self):
        #computes the current sum of squared errors
        csum = 0
        for x, y in self.points:
            csum += (y - (self.m*x +self.b)) ** 2
        return csum/float(len(self.points))


    def plot(self):
        #plotting points
        df = pd.DataFrame(self.points, columns=['x', 'y'])
        x = df['x']
        y = df['y']
        plt.figure()
        plt.subplot(2,2,1)
        plt.plot(x, y, 'go')

        #plotting linear function
        y2 = [self.m * num + self.b for num in x]        
        plt.plot(x,y2)
        
        #display graph
        plt.xlim([x.min()-1, x.max()+1])
        plt.ylim([y.min()-1, y.max()+1])
        plt.show()
