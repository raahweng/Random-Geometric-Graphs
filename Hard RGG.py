import numpy as np
import numpy.random as rn
import matplotlib.pyplot as plt
import networkx as nx
import time
from itertools import product
import scipy.sparse as sp
#from numba import njit, prange
from collections import deque
from torusplot import torusplot
import scipy.sparse.linalg as spla

rho = 250
N = 1000
L = np.sqrt(N/rho)
sigma = 0.05

p = rn.uniform(-L/2, L/2, (N,2))
n = int(np.floor(L / sigma))
l = L / n

cells = np.array((p + L/2) // l, dtype=int)
pairs = [(i, j) for i in range(n) for j in range(n)]
cell_tuples = np.ravel_multi_index((cells[:, 0], cells[:, 1]), (n, n))
inds = {pair: np.flatnonzero(cell_tuples == np.ravel_multi_index(pair, (n, n))) for pair in pairs}
neighbours = {pair: [(i,j) for i in [(pair[0]-1)%n, pair[0], (pair[0]+1)%n] for j in [(pair[1]-1)%n, pair[1], (pair[1]+1)%n]] for pair in pairs}
cells = [tuple(i) for i in cells]


sigma2 = sigma ** 2

visited = np.zeros(N, dtype=bool)
Laps = []
As = []
ind_map = {0:0}

for i in range(N):
    if not visited[i]:
        
        visited[i] = True
        queue = deque([(i,0)])
        component_nodes = []
        row_indices = []
        col_indices = []
        data = []
        c = 0
        old_ind = {i:0}

        while queue:
            node = queue.popleft()
            candidates = np.concatenate([inds[n] for n in neighbours[cells[node[0]]]])
            candidates = np.unique(candidates)

            if len(candidates) > 0:
                Dp = (p[candidates] - p[node[0]] + L/2) % L - L/2
                valid = np.sum(Dp**2, axis=1) < sigma2
                
                for n in candidates[valid]:

                    if not visited[n]:
                        c += 1
                        queue.append((n,c))
                        old_ind[n] = c
                        ind = c
                    else:
                        ind = old_ind[n]
                    visited[n] = True
                    data.append(1)
                    row_indices.append(node[1])
                    col_indices.append(ind)

        ind_map.update(old_ind)
        A = sp.coo_matrix((data, (row_indices, col_indices)), shape=(c+1, c+1)).tocsr()
        sp.coo_matrix.setdiag(A,0)
        As.append(A)
        Lap = sp.csgraph.laplacian(A, normed=False)
        Laps.append(Lap)

start = time.time()
E = []
k = []
for lap in Laps:
    E += list(np.linalg.eigvalsh(lap.toarray()))
    k.append(lap.shape[0])
E = np.array(E)



""" p = p[list(ind_map.keys())]
inds = 0
for A in As:
    if A.shape[0] > 1:
        torusplot(p[inds:inds+A.shape[0]], L, sigma, tor=False)
    inds += A.shape[0] """



