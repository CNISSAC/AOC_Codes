import numpy as np
import matplotlib.pyplot as plt
from plotcov2 import plotcov2

p1 = np.array([[2, 3]])
p1_2=np.array([2, 3])
p1_3 = p1[:, 0]
p2 = np.array([[2, 3], [5,6]])
p3 = np.array([[2, 3], [5,6], [9,10]])
p4 = np.array([[2, 3, 4], [5,6,7], [9,10,8]])
while True:
    print(p1,p2,p3,p4)