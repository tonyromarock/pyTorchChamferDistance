import numpy as np
import torch

from chamfer_distance import ChamferDistance
chamfer_dist = ChamferDistance()

#...
# points and points_reconstructed are [n_points x 3] matrices

points = torch.from_numpy(np.array([ [ [3.,0.,0.], [0.,3.,0.] ],  [ [10.,0.,0.], [0.,10.,0.] ] ])).float()
points_reconstructed = torch.from_numpy(np.array([ [ [5.,0.,0.], [0.,5.,0.] ]  ,[ [9.,0.,0.], [0.,9.,0.] ] ])).float()

dist1, dist2 = chamfer_dist(points, points_reconstructed)
loss = (torch.mean(dist1)) + (torch.mean(dist2))

print(dist1)
print(dist2)
print(loss)
