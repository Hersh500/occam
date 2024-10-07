import numpy as np
import torch

def predict(particles_norm, explore_noise):
    new_particles = particles_norm + torch.randn(*particles_norm.shape) * explore_noise
    return torch.clip(new_particles, min=0, max=1)


def resample_particles_simple(theta_particles, normalized_weights, num=None):
    num_particles = theta_particles.shape[0]
    if num is None:
        num = num_particles
    indices = np.random.choice(num_particles, num, p=normalized_weights, replace=True)
    new_weights = np.ones(num) * 1/num
    return theta_particles[indices], new_weights


def particle_search(eval_fn,
                    num_steps,
                    explore_noise,
                    cost_weights,
                    input_dim,
                    device,
                    particles=None,
                    fixed_inputs=None,
                    num_particles=1024,
                    batch_size=64,
                    sigma_weight=None):

    if particles is None:
        particles = torch.rand(num_particles, input_dim)
    fixed_inputs_batch = fixed_inputs.repeat(num_particles).reshape((num_particles, fixed_inputs.size(-1)))
    full_inputs = torch.cat([particles, fixed_inputs_batch], dim=-1).float().to(device)

    particle_rewards = torch.zeros(num_particles)
    particle_metrics = torch.zeros(num_particles, cost_weights.shape[-1])
    num_tried = 0
    weights_norm = torch.ones(num_particles) * 1/num_particles
    for step in range(num_steps):
        particles_np, weights = resample_particles_simple(particles.cpu().detach().numpy(), weights_norm.cpu().detach().numpy())
        particles = predict(torch.from_numpy(particles_np), explore_noise)
        while num_tried < num_particles:
            losses, ys, vars = eval_fn(full_inputs[num_tried:num_tried+batch_size,:], cost_weights, sigma_weight)
            particle_rewards[num_tried:num_tried+batch_size] = losses
            particle_metrics[num_tried:num_tried+batch_size] = ys
            num_tried += batch_size
        weights_norm = torch.nn.functional.softmax(particle_rewards, dim=-1)  # ensure they are all above 0

    best_particle_idx = torch.argmax(particle_rewards)
    best_particle = particles[best_particle_idx]
    best_particle_metric_scalar = np.dot(cost_weights, particle_metrics[best_particle_idx].cpu().detach().numpy())
    best_particle_reward = particle_rewards[best_particle_idx].cpu().detach().numpy()
    return best_particle, best_particle_reward, best_particle_metric_scalar, particles, particle_metrics, particle_rewards