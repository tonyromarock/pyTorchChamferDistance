import numpy as np
import torch

from chamfer_distance import ChamferDistance
chamfer_dist = ChamferDistance()
#
# sample usage as external in factored3d: 
#   python3  factored3d/external/chamfer/example.py  
#
# points is a [b x n x 3] matrix
# points_reconstructed is a [b x m x 3] matrix
# Notice that m and n do not have to have the same size, but the 
# pcds in the same batch need to be of the same size to make up
# a matrix.

points = torch.from_numpy(np.array([ [ [10.,0.,0.], [0.,10.,0.] ] ])).float()
points_reconstructed = torch.from_numpy(np.array([ [ [9.,0.,0.], [0.,9.,0.], [12.,0.,0.] ] ])).float()

# This loss defintion is close to what is used for FoldingNet
# reconstruction loss. They do not square the L2 distance as 
# we do.

dist1, dist2 = chamfer_dist(points, points_reconstructed)
loss = max((torch.mean(dist1)),(torch.mean(dist2)))

print(dist1)
print(dist2)
print(loss)
