import netket as nk
from netket.graph import Lattice
import numpy as np
import math
import matplotlib.pyplot as plt
import json



'''
#Constructs a rectangular lattice with distinct horizontal and vertical edges:
basis = np.array([
        [1.0,0.0],
        [0.0,0.5],
    ])
custom_edges = [
        (0, 0, [1.0,0.0]),
        (0, 0, [0.0,1.0]),
    ]
g = Lattice(basis_vectors=basis, pbc=False, extent=[4,6],
        custom_edges=custom_edges)
print(g.n_nodes)


g.draw()
'''

N = 4
#hi = nk.hilbert.Spin(s=1/2, N=N)
#g=nk.graph.Chain(length=N, pbc=True)
#g = nk.graph.Lattice(basis_vectors=[[-1,1],[1,0]], extent=[N,N], pbc=False)

g = nk.graph.Honeycomb(extent=[3, 3], pbc=False)

print(g._sites)
g._sites.remove(g._sites[0])
print(g._sites)

print(g.adjacency_list())
print(g.n_edges)
print(g.n_nodes)
g.draw()

'''
class GraphWithDraw(nk.graph.Graph):
    def __init__(self, edges):
        super().__init__(edges=edges)
        self.basis_vectors = np.array([[1, 0], [0, 1]])
        self.extent = np.array([1, 1])
    
    draw = nk.graph._lattice_draw.draw_lattice

g2 = GraphWithDraw(edges=[(1,2), (2,3)])
g2.draw()
'''

#g2 = nk.graph.Lattice(basis_vectors=[[0,1],[1,0]], extent=[N,N], pbc=False)
#g2.draw()
