import numpy as np
from math import ceil
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates
import random
import plotly.graph_objects as go
import sys
import math
from scipy.spatial import distance

def plot_img(img, do_not_use=[0]):
    plt.figure(do_not_use[0])
    do_not_use[0] += 1
    plt.imshow(img)


def get_transformed_pixels_coords(I, H, shift=None):
    ys, xs = np.indices(I.shape[:2]).astype("float64")
    if shift is not None:
        ys += shift[1]
        xs += shift[0]
    ones = np.ones(I.shape[:2])
    coords = np.stack((xs, ys, ones), axis=2)
    coords_H = (H @ coords.reshape(-1, 3).T).T
    coords_H /= coords_H[:, 2, np.newaxis]
    cart_H = coords_H[:, :2]
    
    return cart_H.reshape((*I.shape[:2], 2))

def apply_H_fixed_image_size(I, H, corners):
    h, w = I.shape[:2] # when we convert to np.array it swaps
    
    # corners
    c1 = np.array([1, 1, 1])
    c2 = np.array([w, 1, 1])
    c3 = np.array([1, h, 1])
    c4 = np.array([w, h, 1])
    
    # transformed corners
    Hc1 = H @ c1
    Hc2 = H @ c2
    Hc3 = H @ c3
    Hc4 = H @ c4
    Hc1 = Hc1 / Hc1[2]
    Hc2 = Hc2 / Hc2[2]
    Hc3 = Hc3 / Hc3[2]
    Hc4 = Hc4 / Hc4[2]
    
    xmin = corners[0]
    xmax = corners[1]
    ymin = corners[2]
    ymax = corners[3]

    size_x = ceil(xmax - xmin + 1)
    size_y = ceil(ymax - ymin + 1)
    
    # transform image
    H_inv = np.linalg.inv(H)
    
    out = np.zeros((size_y, size_x, 3))
    shift = (xmin, ymin)
    interpolation_coords = get_transformed_pixels_coords(out, H_inv, shift=shift)
    interpolation_coords[:, :, [0, 1]] = interpolation_coords[:, :, [1, 0]]
    interpolation_coords = np.swapaxes(np.swapaxes(interpolation_coords, 0, 2), 1, 2)
    
    out[:, :, 0] = map_coordinates(I[:, :, 0], interpolation_coords)
    out[:, :, 1] = map_coordinates(I[:, :, 1], interpolation_coords)
    out[:, :, 2] = map_coordinates(I[:, :, 2], interpolation_coords)
    
    return out.astype("uint8")

def Normalise_last_coord(x):
    xn = x  / x[2,:]
    
    return xn


# 4 degrees of freedom (1 scale, 1 angle, 2 translation)
def generate_similarity_transform(scale, angle, tx, ty):

  # Convert the angle to radians
  angle = np.radians(angle)

  # Compute the sine and cosine of the angle
  c, s = np.cos(angle), np.sin(angle)

  # Generate the affine transform matrix
  H = [[scale * c, -scale * s, tx],
       [scale * s, scale * c, ty],
       [0, 0, 1]]

  return H

def generate_normalizing_matrix_T(scale, x_mean, y_mean):
    T = scale * np.array([[1, 0, -x_mean],
                 [0, 1, -y_mean],
                 [0, 0, 1/scale]])
    return T

def normalize_points(points):
    n = points.shape[1]
    x = points[0,:]
    y = points[1,:]
    x_mean = x.mean()
    y_mean = y.mean()
    
    s = (np.sqrt(2)*n) / np.sum(np.sqrt(np.square(x-x_mean) + np.square(y-y_mean)))
    T = generate_normalizing_matrix_T(s, x_mean, y_mean)
    
    points_normalized = T@points
    
    # Uncomment to see that distance to origin is sqrt(2)
    # dist = 0
    # for i in range(n):
    #     dist += distance.euclidean((0,0), (points_normalized[0,i], points_normalized[1,i]))
    # print(dist/n)
    
    return T, points_normalized


def generate_A_i(point1, point2):
    x_normal = point1
    x_prime = point2
    coef0 = [0,0,0]
    coef1 = -x_prime[2]*x_normal
    coef2 = x_prime[1]*x_normal
    coef3 = x_prime[2]*x_normal
    coef4 = -x_prime[0]*x_normal
    
    A_i = np.array([[*coef0, *coef1, *coef2],
         [*coef3, *coef0, *coef4]]) 
    
    return A_i

def DLT_homography(points1, points2):
    # 1. Normalization of points1
    points1_T, points1_normalized = normalize_points(points1)
    
    # 2. Normalization of points2
    points2_T, points2_normalized = normalize_points(points2)
    
    # 3. Apply the DLT algorithm
    n = points1.shape[1]
    A = np.zeros((2*n,9))
    
    for i in range(n):
        A_i = generate_A_i(points1_normalized[:,i], points2_normalized[:,i])
        A[i*2:(i*2)+2,:] = A_i
    
    U,D,V = np.linalg.svd(A)
    h = V.T[:,-1]
    H_tild = h.reshape((3,3))
    
    # 4. Denormalization
    H = np.linalg.inv(points2_T) @ H_tild @ points1_T
    
    return H

def Inliers(H, points1, points2, th):
    # Check that H is invertible
    if abs(math.log(np.linalg.cond(H))) > 15:
        idx = np.empty(1)
        return idx
    
    inliers = []
    n = points1.shape[1]
    
    points2_hat = H @ points1
    points1_hat = np.linalg.inv(H) @ points2
    
    points1_euclid = points1[0:2,:] / points1[2, :]
    points2_euclid = points2[0:2,:] / points2[2, :]
    points1_hat_euclid = points1_hat[0:2,:] / points1_hat[2, :]
    points2_hat_euclid = points2_hat[0:2,:] / points2_hat[2, :]
    
    
    # In case we want to use geometric distance d
    for i in range(n):
        putative_distance = (points1_hat_euclid[0,i]-points1_euclid[0,i])**2 + (points1_hat_euclid[1,i]-points1_euclid[1,i])**2 \
            + (points2_hat_euclid[0,i]-points2_euclid[0,i])**2 + (points2_hat_euclid[1,i]-points2_euclid[1,i])**2 
        
        
        if np.sqrt(putative_distance) < th:
            inliers.append(i)
    
    return np.array(inliers)


def Ransac_DLT_homography(points1, points2, th, max_it):
    Ncoords, Npts = points1.shape
    
    it = 0
    best_inliers = np.empty(1)
    
    while it < max_it:
        indices = random.sample(range(1, Npts), 4)
        H = DLT_homography(points1[:,indices], points2[:,indices])
        inliers = Inliers(H, points1, points2, th)
        
        # test if it is the best model so far
        if inliers.shape[0] > best_inliers.shape[0]:
            best_inliers = inliers
        
        
        # update estimate of iterations (the number of trials) to ensure we pick, with probability p,
        # an initial data set with no outliers
        fracinliers = inliers.shape[0]/Npts
        pNoOutliers = 1 -  fracinliers**4
        eps = sys.float_info.epsilon
        pNoOutliers = max(eps, pNoOutliers)   # avoid division by -Inf
        pNoOutliers = min(1-eps, pNoOutliers) # avoid division by 0
        p = 0.99
        max_it = math.log(1-p)/math.log(pNoOutliers)
        
        it += 1
    
    # compute H from all the inliers
    H = DLT_homography(points1[:,best_inliers], points2[:,best_inliers])
    inliers = best_inliers
    
    return H, inliers



def optical_center(P):
    U, d, Vt = np.linalg.svd(P)
    o = Vt[-1, :3] / Vt[-1, -1]
    return o

def view_direction(P, x):
    # Vector pointing to the viewing direction of a pixel
    # We solve x = P v with v(3) = 0
    v = np.linalg.inv(P[:,:3]) @ np.array([x[0], x[1], 1])
    return v

def plot_camera(P, w, h, fig, legend):
    
    o = optical_center(P)
    scale = 200
    p1 = o + view_direction(P, [0, 0]) * scale
    p2 = o + view_direction(P, [w, 0]) * scale
    p3 = o + view_direction(P, [w, h]) * scale
    p4 = o + view_direction(P, [0, h]) * scale
    
    x = np.array([p1[0], p2[0], o[0], p3[0], p2[0], p3[0], p4[0], p1[0], o[0], p4[0], o[0], (p1[0]+p2[0])/2])
    y = np.array([p1[1], p2[1], o[1], p3[1], p2[1], p3[1], p4[1], p1[1], o[1], p4[1], o[1], (p1[1]+p2[1])/2])
    z = np.array([p1[2], p2[2], o[2], p3[2], p2[2], p3[2], p4[2], p1[2], o[2], p4[2], o[2], (p1[2]+p2[2])/2])
    
    fig.add_trace(go.Scatter3d(x=x, y=z, z=-y, mode='lines',name=legend))
    
    return

def plot_image_origin(w, h, fig, legend):
    p1 = np.array([0, 0, 0])
    p2 = np.array([w, 0, 0])
    p3 = np.array([w, h, 0])
    p4 = np.array([0, h, 0])
    
    x = np.array([p1[0], p2[0], p3[0], p4[0], p1[0]])
    y = np.array([p1[1], p2[1], p3[1], p4[1], p1[1]])
    z = np.array([p1[2], p2[2], p3[2], p4[2], p1[2]])
    
    fig.add_trace(go.Scatter3d(x=x, y=z, z=-y, mode='lines',name=legend))
    
    return
