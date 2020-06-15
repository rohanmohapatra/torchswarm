import time

import torch
import copy

from particle.particle_factory import get_particle_instance
from utils.parameters import SwarmParameters

class SwarmOptimizer:
    def __init__(self, dimensions, swarm_size, swarm_optimizer_type="standard", particle=None, **kwargs):
        self.swarm_size = swarm_size
        if not particle:
            self.particle = get_particle_instance(swarm_optimizer_type)
        else:
            self.particle = particle
        self.max_iterations = kwargs.get('max_iterations') if kwargs.get('max_iterations') else 100
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = kwargs.get("device") if kwargs.get("device") else device
        self.swarm = []
        self.gbest_position = torch.Tensor([0]).to(device)
        self.gbest_particle = None
        self.gbest_value = torch.Tensor([float("inf")]).to(device)
        for i in range(self.swarm_size):
            self.swarm.append(self.particle(dimensions, **kwargs))

    def optimize(self, function):
        self.fitness_function = function

    def run(self, verbosity=True):
        swarm_parameters = SwarmParameters()
        swarm_parameters.r1 = 0
        swarm_parameters.r2 = 0
        # --- Run
        for iteration in range(self.max_iterations):
            tic = time.monotonic()
            # --- Set PBest
            for particle in self.swarm:
                fitness_cadidate = self.fitness_function.evaluate(particle.position)
                if (particle.pbest_value > fitness_cadidate):
                    particle.pbest_value = fitness_cadidate
                    particle.pbest_position = particle.position.clone()
            # --- Set GBest
            for particle in self.swarm:
                best_fitness_cadidate = self.fitness_function.evaluate(particle.position)
                if self.gbest_value > best_fitness_cadidate:
                    self.gbest_value = best_fitness_cadidate
                    self.gbest_position = particle.position.clone()
                    self.gbest_particle = copy.deepcopy(particle)
            r1s = []
            r2s = []
            # --- For Each Particle Update Velocity
            for particle in self.swarm:
                parameters = particle.update_velocity(self.gbest_position)
                particle.move()
                r1s.append(parameters.r1)
                r2s.append(parameters.r2)
            toc = time.monotonic()
            swarm_parameters.r1 = (sum(r1s) / self.swarm_size).item()
            swarm_parameters.r2 = (sum(r2s) / self.swarm_size).item()
            if verbosity == True:
                print('Iteration {:.0f} >> global best fitness {:.3f}  | iteration time {:.3f}'
                      .format(iteration + 1, self.gbest_value.item(), toc - tic))
        swarm_parameters.gbest_position = self.gbest_position
        swarm_parameters.gbest_value = self.gbest_value.item()
        swarm_parameters.c1 = self.gbest_particle.c1
        swarm_parameters.c2 = self.gbest_particle.c2
        return swarm_parameters
