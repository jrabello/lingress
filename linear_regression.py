import pandas as pd
import matplotlib.pyplot as plt
from gradient_decent import *
from ggplot import *
import numpy as np

class LinearRegression:
    """
        data = two features(x,y)
        m = initial slope
        b = initial y-intercept        
        y = b + m*x
    """
    def __init__(self, file_name = '', b = 0, m = 0, n_iter = 100, learning_rate = 0.01):
        """ building dataframe from file and storing features at points attribute """
        self.df = pd.read_csv(file_name)        
        self.points = self.df.iloc[0:100, [0, 2]].values.tolist()
        self.b = b
        self.m = m       
        self.n_iter = n_iter
        self.errors = []
        self.epochs = []
        self.gd = GradientDescent(learning_rate)


    def compute(self):
        """ computing optimal m and b that minimizes the error of cost function"""
        for i in range(0, self.n_iter):
            self.epochs.append(i)
            self.errors.append(self.sse())
            self.b, self.m = self.gd.step(self.points, self.b, self.m)
            #print("{} {} {}".format(self.sse(), self.m, self.b))


    def sse(self):
        """computes the current sum of squared errors(our chosen cost function)"""
        csum = 0
        for x, y in self.points:
            csum += (y - (self.m*x +self.b)) ** 2
        return csum/float(len(self.points))


    def gplot(self):
        """plotting data and linear line"""
        print (ggplot(self.df, aes('SepalLengthCm','PetalLengthCm')) + \
            xlab('sepal length') + ylab('petal length') + \
            geom_point(color='red') + \
            geom_abline(slope=self.m, intercept=self.b, color='steelblue'))
            #xlim(0,2050)            
            #stat_smooth(method='lm') + \

        """plotting errors"""
        df = pd.DataFrame({'epochs':self.epochs,'error':self.errors})
        print (ggplot(df, aes('epochs', 'error')) + \
            geom_point(color='red') + \
            geom_line(color='red'))


    def plot(self):
        """plotting points"""
        df = pd.DataFrame(self.points, columns=['x', 'y'])
        x = df['x']
        y = df['y']
        plt.figure()
        plt.subplot(2,2,1)
        plt.plot(x, y, 'go')

        """plotting linear function"""
        y2 = [self.m * num + self.b for num in x]        
        plt.plot(x,y2)
        
        """display graph"""
        plt.xlim([x.min()-1, x.max()+1])
        plt.ylim([y.min()-1, y.max()+1])
        plt.show()
