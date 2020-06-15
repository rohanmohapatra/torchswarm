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

## Contributors:
- [Rohan Mohapatra](https://github.com/rohanmohapatra)



