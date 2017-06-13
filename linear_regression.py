import pandas as pd
import matplotlib.pyplot as plt
from gradient_decent import *


class LinearRegression:
    """        
        data = two features(x,y) each n-dimentional -> ex:[[1,2],[3,4]]
        w1 = initial slope
        w0 = initial y-intercept        
        y = w0 + w1*x
    """

    def __init__(self, points, w0 = 0, w1 = 0):
        self.w0 = w0
        self.w1 = w1
        self.points = points


    def compute(self):
        gd = GradientDescent()
        for i in range(0, gd.num_iterations):
            self.w0, self.w1 = gd.step(self.points, self.w0, self.w1)
            #print("{} {} {}".format(self.sse(), self.w0, self.w1))


    def sse(self):
        #computes the current sum of squared errors        
        sum = 0
        for x, y in self.points:
            sum += (y - (self.w1*x +self.w0)) ** 2
        return sum/float(len(self.points))


    def plot(self):
        #plotting points
        df = pd.DataFrame(self.points, columns=['x', 'y'])
        x = df['x']
        y = df['y']
        plt.figure()
        plt.subplot(2,2,1)
        plt.plot(x, y, 'go')

        #plotting linear function
        y2 = [self.w1 * num + self.w0 for num in x]        
        plt.plot(x,y2)
        
        #display graph
        plt.xlim([x.min()-1, x.max()+1])
        plt.ylim([y.min()-1, y.max()+1])
        plt.show()
