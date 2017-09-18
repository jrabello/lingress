import numpy as np
from sklearn.preprocessing import StandardScaler

class Feature:
    def __init__(self, name = '', label = '', data = []):
        self.name = name
        self.label = label
        self.data = data
    
    def scale(self):     
        """
        using standard scaler from sklearn, 
        transforming data to make mean zero, 
        so gradient descent converges faster
        """
        return StandardScaler().fit_transform(self.data[:, np.newaxis]).flatten()
        