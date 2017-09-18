from linear_regression import *


def main():    
    # computing linear regression with gradient descent    
    lr = LinearRegression(file_name='housing.data.csv', 
        n_iter=20,
        learning_rate=0.1)   
    print("error: {} y={}x + {}".format(lr.sse(),lr.m,lr.b))
    lr.compute()
    print("error: {} y={}x + {}".format(lr.sse(),lr.m,lr.b))
    lr.gplot()
   

if __name__ == '__main__':
    main()