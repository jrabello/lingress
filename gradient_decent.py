class GradientDescent:
    
    def __init__(self, learning_rate, n_iter):        
        self.n_iter = n_iter
        self.errors = []
        self.epochs = range(1, n_iter+1)
        self.learning_rate = learning_rate


    def step(self, points, b, m):
        """
            computes the error relative to each variable,
            and tries to minimize the error of our cost function(sse) 
            by finding a local(or global) minimum
        """
        b_gradient = 0
        m_gradient = 0
        N = float(len(points))

        for x,y in points:
            b_gradient += -(2/N) * (y - ((m * x) + b))
            m_gradient += -(2/N) * x * (y - ((m * x) + b))
            #print('m_grad:{}'.format(m_gradient))
        
        b -= (b_gradient * self.learning_rate)
        m -= (m_gradient * self.learning_rate)
        return b, m
    

