from linear_regression import *


def main():    
    # computing linear regression with gradient descent
    lr = LinearRegression(file='hr_year.csv', iterations=10, learning_rate=0.0000001)
    print("error: {} m:{} b:{}".format(lr.sse(), lr.m, lr.b))
    lr.compute()
    print("error: {} m:{} b:{}".format(lr.sse(), lr.m, lr.b))
    lr.gplot()
    

if __name__ == '__main__':
    main()
