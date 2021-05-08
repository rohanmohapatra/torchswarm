from torchswarm.particle import Particle, RotatedParticle, ExponentiallyWeightedMomentumParticle, RotatedEWMParticle


def get_particle_instance(swarm_optimizer_type):
    if swarm_optimizer_type == "standard":
        return Particle
    if swarm_optimizer_type == "rotated":
        return RotatedParticle
    if swarm_optimizer_type == "exponentially_weighted":
        return ExponentiallyWeightedMomentumParticle
    if swarm_optimizer_type == "rotated_exponentially_weighted":
        return RotatedEWMParticle
