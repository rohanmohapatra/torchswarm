import unittest

import torch

from particle import Particle, ExponentiallyWeightedMomentumParticle
from swarmoptimizer import SwarmOptimizer
from utils.parameters import SwarmParameters
from utils.rotation_utils import get_rotation_matrix, get_inverse_matrix, get_phi_matrix
import numpy as np

class SwarmOptimizerTest(unittest.TestCase):
    def setUp(self) -> None:
        class TestFunction1:
            def evaluate(self, x):
                return x ** 2 + torch.exp(x)

        class TestFunction2:
            def __init__(self, y):
                self.y = y
                self.fitness = torch.nn.BCELoss()

            def evaluate(self, x):
                return self.fitness(x, self.y)

        class CustomParticle(ExponentiallyWeightedMomentumParticle):
            def __init__(self, dimensions, beta=0.9, c1=2, c2=2, **kwargs):
                super(CustomParticle, self).__init__(dimensions, beta, c1, c2, **kwargs)

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

        self.test_function_1 = TestFunction1
        self.test_function_2 = TestFunction2
        self.bring_your_particle = CustomParticle
    def test_standard_pso(self):
        pso = SwarmOptimizer(1, 100, max_iterations=10)
        pso.optimize(self.test_function_1())
        results = pso.run(verbosity=True)
        self.assertAlmostEqual(results.gbest_value, 0.827, 3)

    def test_rotated_em_pso(self):
        true_y = torch.randint(0, 2, (4, 4)).type(torch.FloatTensor)
        pso = SwarmOptimizer(4, 100, swarm_optimizer_type="rotated_exponentially_weighted", max_iterations=20, classes=4, bounds=[0,1])
        pso.optimize(self.test_function_2(true_y))
        results = pso.run(verbosity=True)
        self.assertAlmostEqual(results.gbest_value, 0.0, 3)

    def test_bring_your_own_particle(self):
        true_y = torch.randint(0, 2, (4, 4)).type(torch.FloatTensor)
        pso = SwarmOptimizer(4, 100, particle=self.bring_your_particle, max_iterations=20,
                             classes=4, bounds=[0, 1])
        pso.optimize(self.test_function_2(true_y))
        results = pso.run(verbosity=True)
        self.assertAlmostEqual(results.gbest_value, 0.0, 3)

if __name__ == '__main__':
    unittest.main()
