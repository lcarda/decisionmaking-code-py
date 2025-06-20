import sys; sys.path.append('./src/'); sys.path.append('../'); sys.path.append('../../')

import numpy as np

from ch24 import SimpleGame

class TravelersDilemma(SimpleGame):
    def __init__(self,P=2,V=100):
        self.P = P
        self.V = V
        gamma = 0.9        
        I = [0, 1]  # two agents
        A = [[i for i in range(2, self.V+1, 1)], [i for i in range(2, self.V+1, 1)]]  # $2 - $100
        def R(a):
            if a[0] == a[1]:
                return np.array([a[0], a[1]])
            elif a[0] < a[1]:
                return np.array([a[0] + self.P, a[0] - self.P])
            else:  # a[0] > a[1]
                return np.array([a[1] - self.P, a[1] + self.P])

        super().__init__(gamma, I, A, R)
