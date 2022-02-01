import numpy as np
import scipy.linalg

class MultivariateNormal:
    def __init__(self, covariance : np.array):
        self.L = scipy.linalg.cholesky(covariance, lower=True)
    
    def sample(self):
        return self.L @ np.random.normal(size=(self.L.shape[0], 1))
