from linear_regression import *


def main():    
    # computing linear regression with gradient descent
    data = [ [1,3], [2,1], [3,4], [4,2], [5,6] ]
    lr = LinearRegression(data, iterations=1000, learning_rate=0.01)

    print("error: {} m:{} b:{}".format(lr.sse(), lr.m, lr.b))
    lr.compute()
    print("error: {} m:{} b:{}".format(lr.sse(), lr.m, lr.b))
    lr.plot()


if __name__ == '__main__':
    main()
