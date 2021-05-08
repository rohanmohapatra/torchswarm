# Integration Test
import torch

from torchswarm.swarmoptimizer import SwarmOptimizer


class CubicFunction:
    def evaluate(self, x):
        return x ** 2 + torch.exp(x)

empso = SwarmOptimizer(1, 100, swarm_optimizer_type="exponentially_weighted", max_iterations=10)
empso.optimize(CubicFunction())

print(empso.run(verbosity=True).__dict__)