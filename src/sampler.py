import torch


class Sampler:
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_schedule = self.linear_beta_schedule()
        self.alpha = 1 - self.beta_schedule
        self.alpha_cummulative_prod = torch.cumprod(self.alpha, dim=-1)

    def linear_beta_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.num_timesteps)

    def _repeated_unsqueeze(self, target, tensor):
        while target.dim() > tensor.dim():
            tensor = tensor.unsqueeze(-1)
        return tensor

    def add_noise(self, images, timesteps):
        batch_size, c, h, w = images.shape
        device = images.device
        alpha_cummulative_prod_timesteps = self.alpha_cummulative_prod[timesteps.cpu()].to(device)
        mean_coeff = alpha_cummulative_prod_timesteps**0.5
        var_coeff = (1 - alpha_cummulative_prod_timesteps) ** 0.5
        mean_coeff = self._repeated_unsqueeze(images, mean_coeff)
        var_coeff = self._repeated_unsqueeze(images, var_coeff)
        noise = torch.randn_like(images)
        """print(mean_coeff.shape)
        print(image.shape)"""
        noisy_image = mean_coeff * images + var_coeff * noise
        return noisy_image, noise

    def remove_noise(self, images, timesteps, predicted_noise):
        b, c, h, w = images.shape
        device = images.device
        equal_to_zero_mask = timesteps == 0
        beta_t = self.beta_schedule[timesteps.cpu()].to(device)
        alpha_t = self.alpha[timesteps.cpu()].to(device)
        alpha_cummulative_prod_t = self.alpha_cummulative_prod[timesteps.cpu()].to(device)
        alpha_cummulative_prod_t_prev = self.alpha_cummulative_prod[(timesteps.cpu() - 1)].to(device)
        alpha_cummulative_prod_t_prev[equal_to_zero_mask] = (
            1.0  # @QUESTION: this line of code looks weird
        )
        noise = torch.randn_like(images)  # This is element z in line 4 in Algorithm 2 Sampling
        variance = (
            beta_t * (1 - alpha_cummulative_prod_t_prev) / (1 - alpha_cummulative_prod_t)
        )  # This is element beta_t_hat in formula (7)
        variance = self._repeated_unsqueeze(images, variance)
        sigma_t_z = (
            variance**0.5
        ) * noise  # This is element sigma * z in line 4 in Algorithm 2 Sampling
        noise_coff = (
            beta_t / (1 - alpha_cummulative_prod_t) ** 0.5
        )  # This is an element in line 4 in Algorithm 2 Sampling, in the paper, they write beta_t in form of (1 - alpha_t)
        noise_coff = self._repeated_unsqueeze(images, noise_coff)
        reciprocal_root_alpha_t = alpha_t ** (
            -0.5
        )  # This is the first element in Algorithm 2 Sampling
        reciprocal_root_alpha_t = self._repeated_unsqueeze(images, reciprocal_root_alpha_t)

        # Final formula in Algorithm 2 Sampling
        mean = reciprocal_root_alpha_t * (images - noise_coff * predicted_noise)
        denoised = mean + sigma_t_z

        return denoised
