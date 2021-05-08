# Integration Test
import torch

from swarmoptimizer import SwarmOptimizer


class CubicFunction:
    def evaluate(self, x):
        return x ** 2 + torch.exp(x)

pso = SwarmOptimizer(1, 100, swarm_optimizer_type="exponentially_weighted", max_iterations=10)
pso.optimize(CubicFunction())

print(pso.run(verbosity=True).__dict__)