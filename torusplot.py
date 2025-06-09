import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import s3dlib.surface as s3d

def torusFunc(rtz) :
    r,t,z = rtz
    Z = np.sin(z*np.pi)/3
    R = r + np.cos(z*np.pi)/3
    return R,t,Z

def torusplot(p, L, sigma, s, tor=False, soft=False):

    N = p.shape[0]
    extended_p = []
    shifts = [(0, 0), (L, 0), (0, L), (-L, 0), (0, -L), (L, L), (-L, L), (L, -L), (-L, -L)]
    for dx, dy in shifts:
        shifted_p = p + np.array([dx, dy])
        extended_p.append(shifted_p)
    p = np.vstack(extended_p)
    p = p[np.max(np.abs(p), axis=1) < L/2 + sigma]
    N = p.shape[0]

    D = np.repeat(p[:, :, np.newaxis], N, axis=2).transpose(0,2,1)
    Ds = D - D.transpose(1,0,2)
    Df = (Ds[:,:,0] ** 2 + Ds[:,:,1] ** 2) ** 0.5

    if soft:
        A = -1/(2*np.pi*sigma**2) * np.exp(-Df / (2*sigma**2))
        np.fill_diagonal(A, 0)
        G = nx.from_numpy_array(-A)

        edges = G.edges()
        weights = [np.abs(G[u][v]['weight']) for u,v in edges]
        points = [(p[i,0], p[i,1]) for i in range(N)]
        pos = {i: point for i,point in enumerate(points)}
        plt.figure(figsize=(15, 15), dpi=50)
        nx.draw(G, pos=pos, width=100*weights/np.max(weights), node_size=s)
        plt.xlim([-L/2, L/2])
        plt.ylim([-L/2, L/2])
        plt.show()
    else:

        A = 1 - Df / sigma
        A[A <= 0] = 0
        A[A > 0] = 1
        np.fill_diagonal(A, 0)

        G = nx.from_numpy_array(A)
        edges = G.edges()
        weights = [np.abs(G[u][v]['weight']) for u,v in edges]
        points = [(p[i,0], p[i,1]) for i in range(N)]
        pos = {i: point for i,point in enumerate(points)}

        if not tor:
            plt.figure(figsize=(15, 15), dpi=50)
            nx.draw(G, pos=pos, width=10*weights/np.max(weights), node_size=s)
            plt.xlim([-L/2, L/2])
            plt.ylim([-L/2, L/2])
        else:
            plt.figure(figsize=(10, 10), dpi=50)
            nx.draw(G, pos=pos, width=10*weights/np.max(weights), node_size=s)
            plt.xlim([-L/2, L/2])
            plt.ylim([-L/2, L/2])
            plt.savefig('temp.png')
            img_fname, rez = 'temp.png', 6
            
            torus = s3d.CylindricalSurface(rez, basetype='squ_s')
            torus.map_color_from_image(img_fname)
            torus.map_geom_from_op(  torusFunc) 

            minmax = (-1.5,1.5)
            fig = plt.figure(figsize=plt.figaspect(1))
            ax = plt.axes(projection='3d')
            ax.set(xlim=minmax, ylim=minmax, zlim=minmax)
            ax.set_title("torus")
            ax.set_axis_off()
            ax.view_init(elev=20, azim=-125)
            ax.add_collection3d(torus.shade(.5).hilite())

        plt.show()
