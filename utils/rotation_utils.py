import torch
import numpy as np
import random
def get_rotation_matrix(dimensions, theta, axis_rotation_factor):
    matrix = np.zeros((dimensions, dimensions))
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    dimensions_list = list(range(1, dimensions+1))
    selected = random.sample(dimensions_list, 2*int(axis_rotation_factor*dimensions))
    non_selected = [x for x in dimensions_list if x not in selected]
    selected = np.asarray(sorted(selected))
    for i in range(0, selected.size, 2):
        idx = selected[i]
        next_idx = selected[i+1]
        matrix[idx-1][idx-1] = cos_theta
        matrix[idx-1][next_idx-1] = -sin_theta
        matrix[next_idx-1][idx-1] = sin_theta
        matrix[next_idx-1][next_idx-1] = cos_theta
    for i in range(0, len(non_selected)):
        idx = non_selected[i]
        matrix[idx-1][idx-1] = 1
    return torch.from_numpy(matrix)

def get_phi_matrix(dimensions, c, r):
    matrix = np.zeros((dimensions, dimensions))
    for i in range(dimensions):
        matrix[i][i] = c*r
    return torch.from_numpy(matrix)

def get_inverse_matrix(input):
    return torch.inverse(input)
