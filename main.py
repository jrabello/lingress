from linear_regression import *


def main():    
    # computing linear regression with gradient descent    
    lr = LinearRegression(file_name='iris.data', n_iter=10, learning_rate=0.01)
    print("error: {} m:{} b:{}".format(lr.sse(), lr.m, lr.b))
    lr.compute()
    print("error: {} m:{} b:{}".format(lr.sse(), lr.m, lr.b))
    lr.gplot()
    

if __name__ == '__main__':
    main()
