# Torchswarm

A fast implementation of Particle Swarm Optimization using PyTorch

## We support

### Variants of Particle Swarm Optimization

We support for all kinds of PSO

### Bring your own particle

We allow for getting a custom particle with a different velocity update rule,
The Class must have the following methods:

- `__init__`
- `move`
- `update_velocity`

## How to define your problem.

Create a class by inheriting `torchswarm.functions.Function` and an `evaluate` method.

```python
class XSquare(Function):
    def evaluate(self, x):
        return x**2
```

## Example

```python
import torch

from torchswarm.swarmoptimizer import SwarmOptimizer


class CubicFunction:
    def evaluate(self, x):
        return x ** 2 + torch.exp(x)

empso = SwarmOptimizer(1, 100, swarm_optimizer_type="exponentially_weighted", max_iterations=10)
empso.optimize(CubicFunction())

print(empso.run(verbosity=True).__dict__)
```

## Contributors:

- [Rohan Mohapatra](https://github.com/rohanmohapatra)
