import numpy as np
import torch

from chamfer_distance import ChamferDistance
chamfer_dist = ChamferDistance()

#...
# points and points_reconstructed are [n_points x 3] matrices

points = torch.from_numpy(np.array([ [10,0,0], [0,10,0], [0,0,10] ]))
points_reconstructed = torch.from_numpy(np.array( [ [9,0,0], [0,9,0], [0,0,9] ] ))

dist1, dist2 = chamfer_dist(points, points_reconstructed)
loss = (torch.mean(dist1)) + (torch.mean(dist2))

print(dist1)
print(dist2)