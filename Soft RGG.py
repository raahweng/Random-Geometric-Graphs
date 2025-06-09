import numpy as np
import numpy.random as rn
import matplotlib.pyplot as plt
import networkx as nx
import time
from scipy.linalg import pinv
from torusplot import torusplot

plt.style.use("seaborn-v0_8-white")
plt.rcParams.update({'font.size': 15})
plt.rcParams['axes.facecolor']='w'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.edgecolor'] = plt.rcParams['axes.labelcolor']

rho = 250
N = 10000
L = np.sqrt(N/rho)

start = time.time()
Elist = []
p = rn.uniform(-L/2, L/2, (N,2))

D = np.repeat(p[:, :, np.newaxis], N, axis=2).transpose(0,2,1)
Ds = D - D.transpose(1,0,2)
Dp = (Ds + L/2) % L - L/2
Df = np.sqrt(Dp[:,:,0] ** 2 + Dp[:,:,1] ** 2)

sigma = 0.0758
M1 = - np.exp(-Df / (2*sigma**2))
np.fill_diagonal(M1, 0)
np.fill_diagonal(M1, -np.sum(M1, axis=0))

torusplot(p, L, sigma, 10, tor=False, soft=True)


