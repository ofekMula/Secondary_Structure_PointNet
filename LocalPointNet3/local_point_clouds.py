from numba import njit, prange
import numpy as np
# If numba package is problematic: remove the decorators @njit, replace prange by range. Will be slower but will run fine.
curr_float = np.float64
curr_int = np.int64




'''
Compute pairwise distances between list of points.
Input = N X d coordinate matrix.
Output = N X N symmetric distance matrix.
'''
@njit(parallel=True)
def _distancesnumba2(x):
    N = x.shape[0]
    D = np.zeros((N, N), dtype=curr_float)
    for n1 in prange(N):
        D[n1, n1] = 0
        for n2 in prange(n1 + 1, N):
            D[n1, n2] = np.sqrt(((x[n1] - x[n2])**2).sum())
            D[n2, n1] = D[n1, n2]
    return D

'''
Compute vector product between a pair of list of points
Input: (c1 = N X 3 coordinate matrix, c2= N X 3 coordinate matrix)
Ouput: elementwise vector product c1X c2
'''
@njit(parallel=True)
def _vector_product2(x1, x2, normalized=False):
    out = np.empty_like(x1)
    out[:, 0] = x1[:, 1] * x2[:, 2] - x1[:, 2] * x2[:, 1]
    out[:, 1] = x1[:, 2] * x2[:, 0] - x1[:, 0] * x2[:, 2]
    out[:, 2] = x1[:, 0] * x2[:, 1] - x1[:, 1] * x2[:, 0]
    if normalized:
        for n in range(out.shape[0]):
            out[n] /= np.sqrt((out[n]**2).sum())
    return out



'''
Code for generating local frames from a backbone coordinates. 

- Input: Series of 3D coordinates LX3 (e.g. list of Calphas) 
- Output: A list of frames, size  L X 4 X 3. Each frame is a 4X 3 matrix, the first row is the origin of the frames, the the second to fourth row are the three normed vectors (x’,y’,z’).  
- Algorithm: 
a) add virtual points (for boundary conditions): 
P_-1 = P_0 - (P_2-P_1)
P_L = P_{L-1} + (P_{L-1}-P_{L-2})
b) For each point i in 0:L-1:
- take the triplet of points (P_{i-1}, P_i, P_{i+1}). 
- Define origin as P_i
- Define first vector u_x' = (P_{i+1} - P_{i})
- Define third vector u_z' = (P_{i} - P_{i-1}) X u_x' (= the normal of the plane containing P_{i-1},P_{i},P_{i+1})
- Define second vector y' = u_z' X u_x' (such that the frame )
- Normalize (u_x',u_y',u_z') to unit norm.
'''
@njit(parallel=True)
def build_frame(x):
    N = x.shape[0]
    delta_x = np.zeros((N + 1, 3), dtype=curr_float)
    delta_x[1:-1, :] = (x[1:, :] - x[:-1, :])
    delta_x[0, :] = 1.0 * delta_x[2, :]
    delta_x[-1, :] = 1.0 * delta_x[-3, :]
    for n in prange(N + 1):
        delta_x[n] /= np.sqrt((delta_x[n]**2).sum())

    normal = _vector_product2(
        delta_x[:-1, :], delta_x[1:, :], normalized=True)
    normal_cross_delta_x = _vector_product2(
        normal[:, :], delta_x[1:, :])
    delta_x = delta_x[1:, :]

    frame = np.concatenate(
        (
            np.expand_dims(x, 1),
            np.expand_dims(delta_x, 1),
            np.expand_dims(normal_cross_delta_x, 1),
            np.expand_dims(normal, 1)
        ), axis=1)
    return frame



'''
2. Code for generating local point clouds from local frames.
- Input: A list of points LX3, a list of frames L X 4 X 3, a neighborhood size K.
- Output: A list of local points clouds, of size L X K X 3.
- Algorithm:
For each frame (center, u_x',u_y',u_z')
- Find the K closest points, with coordinates (x_k,y_k,z_k).
- Compute their local coordinates:
x'_k = < (x_k - center), u_x' >
y'_k = < (y_k - center), u_y' >
z'_k = < (z_k - center), u_z' >
Group all results into a 3D array.
'''
@njit(parallel=True)
def build_local_point_cloud(x, K=10):
    N = x.shape[0]
    xlocal = np.zeros((N, K, 3), dtype=curr_float)
    neighbors = np.zeros((N, K), dtype=curr_int)
    D = _distancesnumba2(x)
    for n in prange(N):
        neighbors[n] = np.argsort(D[n])[:K]
    frames = build_frame(x)
    xrelative = np.zeros((K, 3), dtype=curr_float)
    for n in prange(N):
        for k in range(K):
            xrelative[k, :] = x[neighbors[n, k], :] - frames[n,0, :]
        xlocal[n, :, 0] = np.dot(xrelative, frames[n,1, :])
        xlocal[n, :, 1] = np.dot(xrelative, frames[n,2, :])
        xlocal[n, :, 2] = np.dot(xrelative, frames[n,3, :])
    return xlocal, neighbors


