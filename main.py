from linear_regression import *


def main():    
    # computing linear regression with gradient descent
    data = [ [1,3], [2,1], [3,4], [4,2], [5,6] ]
    lr = LinearRegression(data)
    print("error: {} w1:{} w0:{}".format(lr.sse(), lr.w1, lr.w0))
    lr.compute()
    print("error: {} w1:{} w0:{}".format(lr.sse(), lr.w1, lr.w0))
    lr.plot()


if __name__ == '__main__':
    main()
