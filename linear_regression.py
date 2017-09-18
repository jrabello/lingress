import pandas as pd
import matplotlib.pyplot as plt
from gradient_decent import *
from ggplot import *
import numpy as np
from feature import *

class LinearRegression:
    """
        data = two features(x,y)
        w1 = initial slope(m)
        w0 = initial y-intercept(b)
        x2 = w0 + w1*x
    """
    def __init__(self, file_name = '', b = 0, m = 0, n_iter = 100, learning_rate = 0.01):        
        """ 
        building dataframe from file and storing features at points attribute        
        choosing two features RM(average number of rooms per dwelling) and 
        MEDV(Median value of owner-occupied homes in $1000's)
        because correlation coeficient(x_covariance/x1_std*x2_std) is 0.70
        """ 
        self.b = b
        self.m = m           
        self.points = []
        self.df = pd.read_csv(file_name, header=None, sep='\s+')
        self.df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 
              'NOX', 'RM', 'AGE', 'DIS',  
              'RAD', 'TAX', 'PTRATIO', 'B', 
              'LSTAT', 'MEDV']
        self.feature_x1 = Feature(name='RM', data=self.df['RM'], label='average number of rooms (std)')
        self.feature_x2 = Feature(name='MEDV', data=self.df['MEDV'], label='price in $1000\'s (std)') 
        self.df['RM'] = self.feature_x1.scale()
        self.df['MEDV'] = self.feature_x2.scale()
        for x,y in zip(self.df['RM'].values, self.df['MEDV'].values):
            self.points.append([x,y])        
        self.gd = GradientDescent(n_iter = n_iter, learning_rate=learning_rate)


    def compute(self):
        # computing optimal m and b that minimizes the error of cost function
        for i in range(0, self.gd.n_iter):
            #self.gd.epochs.append(i)
            self.gd.errors.append(self.sse())
            self.b, self.m = self.gd.step(self.points, self.b, self.m)
            #print("{} {} {}".format(self.sse(), self.m, self.b))
        

    def sse(self):
        # computes the current sum of squared errors(our cost function)
        csum = 0
        for x, y in self.points:
            output = self.m*x +self.b
            csum += (y - output) ** 2
        return csum/float(len(self.points)) 
       

    def gplot(self):
        # plotting data and linear function
        #print(self.df.head())
        print (ggplot(self.df, aes(self.feature_x1.name, self.feature_x2.name)) + \
            xlab(self.feature_x1.label) + \
            ylab(self.feature_x2.label) + \
            geom_point(color='red') + \
            geom_abline(slope=self.m, intercept=self.b, color='steelblue'))
            #xlim(0,2050)            
            #stat_smooth(method='lm') + \

        # plotting errors        
        df = pd.DataFrame({'epochs':self.gd.epochs,'error':self.gd.errors})
        print (ggplot(df, aes('epochs', 'error')) + \
            geom_point(color='red') + \
            geom_line(color='red'))


    def plot(self):
        # plotting points
        df = pd.DataFrame(self.points, columns=['x', 'y'])
        x = df['x']
        y = df['y']
        plt.figure()
        plt.subplot(2,2,1)
        plt.plot(x, y, 'go')

        # plotting linear function
        y2 = [self.m * num + self.b for num in x]        
        plt.plot(x,y2)
        
        # display graph
        plt.xlim([x.min()-1, x.max()+1])
        plt.ylim([y.min()-1, y.max()+1])
        plt.show()
