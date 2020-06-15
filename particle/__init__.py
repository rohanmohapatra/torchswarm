import torch
import numpy as np

from utils.rotation_utils import get_rotation_matrix, get_inverse_matrix, get_phi_matrix
from utils.parameters import SwarmParameters


class Particle:
    def __init__(self, dimensions, w=0.5, c1=2, c2=2, **kwargs):
        self.dimensions = dimensions
        self.w = w
        self.c1 = c1
        self.c2 = c2
        classes = kwargs.get("classes") if kwargs.get("classes") else 1
        if kwargs.get("bounds"):
            self.bounds = kwargs.get("bounds")
            self.position = (self.bounds[0] - self.bounds[1]) * torch.rand(dimensions, classes) + self.bounds[1]
        else:
            self.bounds = None
            self.position = torch.rand(dimensions, classes)
        self.velocity = torch.zeros((dimensions, classes))
        self.pbest_position = self.position
        self.pbest_value = torch.Tensor([float("inf")])

    def update_velocity(self, gbest_position):
        r1 = torch.rand(1)
        r2 = torch.rand(1)
        for i in range(0, self.dimensions):
            self.velocity[i] = self.w * self.velocity[i] \
                               + self.c1 * r1 * (self.pbest_position[i] - self.position[i]) \
                               + self.c2 * r2 * (gbest_position[i] - self.position[i])

        swarm_parameters = SwarmParameters()
        swarm_parameters.r1 = r1
        swarm_parameters.r2 = r2
        return swarm_parameters

    def move(self):
        for i in range(0, self.dimensions):
            self.position[i] = self.position[i] + self.velocity[i]
        if self.bounds:
            self.position = torch.clamp(self.position, self.bounds[0], self.bounds[1])


class RotatedParticle(Particle):
    def __init__(self, dimensions, w, c1=2, c2=2, **kwargs):
        super(RotatedParticle, self).__init__(dimensions, w, c1, c2, **kwargs)

    def update_velocity(self, gbest_position):
        r1 = torch.rand(1)
        r2 = torch.rand(1)
        a_matrix = get_rotation_matrix(self.dimensions, np.pi / 5, 0.4)
        a_inverse_matrix = get_inverse_matrix(a_matrix)
        x = a_inverse_matrix * get_phi_matrix(self.dimensions, self.c1, r1) * a_matrix
        self.velocity = self.w * self.velocity \
                        + torch.matmul(
            (a_inverse_matrix * get_phi_matrix(self.dimensions, self.c1, r1) * a_matrix).float(),
            (self.pbest_position - self.position).float()) \
                        + torch.matmul(
            (a_inverse_matrix * get_phi_matrix(self.dimensions, self.c2, r2) * a_matrix).float(),
            (gbest_position - self.position).float())
        swarm_parameters = SwarmParameters()
        swarm_parameters.r1 = r1
        swarm_parameters.r2 = r2
        return swarm_parameters


class ExponentiallyWeightedMomentumParticle(Particle):
    def __init__(self, dimensions, beta=0.9, c1=2, c2=2, **kwargs):
        super(ExponentiallyWeightedMomentumParticle, self).__init__(dimensions, 0, c1, c2, **kwargs)
        self.beta = beta
        self.momentum = torch.zeros((dimensions, 1))

    def update_velocity(self, gbest_position):
        r1 = torch.rand(1)
        r2 = torch.rand(1)
        for i in range(0, self.dimensions):
            momentum_t = self.beta * self.momentum[i] + (1 - self.beta) * self.velocity[i]
            self.velocity[i] = momentum_t \
                               + self.c1 * r1 * (self.pbest_position[i] - self.position[i]) \
                               + self.c2 * r2 * (gbest_position[i] - self.position[i])
            self.momentum[i] = momentum_t
        swarm_parameters = SwarmParameters()
        swarm_parameters.r1 = r1
        swarm_parameters.r2 = r2
        return swarm_parameters


class RotatedEWMParticle(ExponentiallyWeightedMomentumParticle):
    def __init__(self, dimensions, beta=0.9, c1=2, c2=2, **kwargs):
        super(RotatedEWMParticle, self).__init__(dimensions, beta, c1, c2, **kwargs)

    def update_velocity(self, gbest_position):
        r1 = torch.rand(1)
        r2 = torch.rand(1)
        momentum_t = self.beta * self.momentum + (1 - self.beta) * self.velocity
        a_matrix = get_rotation_matrix(self.dimensions, np.pi / 5, 0.4)
        a_inverse_matrix = get_inverse_matrix(a_matrix)
        x = a_inverse_matrix * get_phi_matrix(self.dimensions, self.c1, r1) * a_matrix
        self.velocity = momentum_t \
                        + torch.matmul(
            (a_inverse_matrix * get_phi_matrix(self.dimensions, self.c1, r1) * a_matrix).float(),
            (self.pbest_position - self.position).float()) \
                        + torch.matmul(
            (a_inverse_matrix * get_phi_matrix(self.dimensions, self.c2, r2) * a_matrix).float(),
            (gbest_position - self.position).float())

        swarm_parameters = SwarmParameters()
        swarm_parameters.r1 = r1
        swarm_parameters.r2 = r2
        return swarm_parameters
