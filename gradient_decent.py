class GradientDescent:


    def __init__(self, learning_rate = 0.01, num_iterations = 1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations


    def step(self, points, w0, w1):
        """
            computes the error relative to each w,
            and tries to minimize the error finding a local(or global) minimum
        """
        
        w0_gradient = 0
        w1_gradient = 0
        N = float(len(points))

        for x,y in points:
            w0_gradient += -(2/N) * (y - ((w1 * x) + w0))
            w1_gradient += -(2/N) * x * (y - ((w1 * x) + w0))

        w0 = w0 - (w0_gradient * self.learning_rate)
        w1 = w1 - (w1_gradient * self.learning_rate)
        return w0, w1

