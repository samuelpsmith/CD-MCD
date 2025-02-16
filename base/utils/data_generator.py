import numpy as np
def generate_gaussians(x, gaussian_funct, num_gaussians, noise_level, seed, min_distance):
    y = np.zeros_like(x)
    gaussians = []  # Store each Gaussian component
    centers = []  # To track centers of generated Gaussians
    np.random.seed(seed)  # For reproducibility

    for _ in range(num_gaussians):
        # Generate a random center and ensure it's at least `min_distance` away from existing centers
        while True:
            center = np.random.uniform(x.min(), x.max())
            if all(abs(center - c) >= min_distance for c in centers):
                centers.append(center)
                break

        # Generate random amplitude and width for the Gaussian
        amplitude = np.random.uniform(0.5, 1.5)
        width = np.random.uniform(1, 3)

        # Create the Gaussian and add it to the total signal
        g = gaussian_funct(x, amplitude, center, width)
        gaussians.append(g)
        y += g

    # Add white noise
    y += np.random.normal(0, noise_level, size=x.shape)

    return y, gaussians