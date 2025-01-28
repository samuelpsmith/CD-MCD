import numpy as np
from MCD_process.base.constants import tiny, s2pi



def stable_gaussian_sigma(x, amplitude, center, sigma):
    """Returns a numerically stable gaussian. Takes arguments in terms of sigma.
    In the case that non-constant values are found (i.e. not Tiny), then should be normalized to 1.
    Some choices are not clear - like multplying by 4.0 instead of 4.
    We choose not to provide default values because we wish to fail on not passing good values."""
    gamma = sigma * 2 * np.sqrt(2 * np.log(2))
    a_0 = amplitude
    mu = center
    return a_0 * ((2 * np.sqrt(np.log(2))) / max(tiny, (gamma * np.sqrt(np.pi)))) * np.exp(
        -((4.0 * max(tiny, (np.log(2)) / (gamma ** 2)))) * (
                    x - mu) ** 2)  # here we multiply by 4.0 for the same reason that they multiply by 1.0. Maybe this sets as a float.


def stable_gaussian_derivative_sigma(x, amplitude, center, sigma):
    "Returns the normalized first derivative of a gaussian function. Normalized such that f(x) * x = -1"
    gamma = sigma * 2 * np.sqrt(2 * np.log(2))
    a_0 = amplitude
    mu = center
    # return ((- a_0 * 4 * np.log(2) * (x - mu))/ max(tiny, gamma**2)) * np.exp(-((4.0 * max(tiny, (np.log(2))/(gamma**2)))) * (x - mu)**2)
    return (a_0 * -16.0 * np.log(2) * np.sqrt(np.log(2)) * (x - mu)) / (
        max(tiny, (gamma ** 3 * np.sqrt(np.pi)))) * np.exp(-((4.0 * max(tiny, (np.log(2)) / (gamma ** 2)))) * (
                x - mu) ** 2)  # this function is not normalized to be area under curve of 1.


def custom_gaussian_old(x, amplitude, center, sigma):
    """Gaussian function."""
    return (amplitude / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((x - center) ** 2) / (2 * sigma ** 2))


def custom_gaussian(x, amplitude=1.0, center=0.0, sigma=1.0):
    """Return a 1-dimensional Gaussian function.

    gaussian(x, amplitude, center, sigma) =
        (amplitude/(s2pi*sigma)) * exp(-(1.0*x-center)**2 / (2*sigma**2))

    """
    return ((amplitude / (max(tiny, s2pi * sigma)))
            * np.exp(-(1.0 * x - center) ** 2 / max(tiny, (
                        2 * sigma ** 2))))  # here, why do they multiply by 1.0? Is it to set it to a float always?


def gaussian_old(x, amplitude, center, width):
    return amplitude * np.exp(-(x - center) ** 2 / (2 * width ** 2))


def gaussian(x, mu, gamma, a_0):
    "Returns a normalized gaussian function. Normalized to 1"
    return a_0 * ((2 * np.sqrt(np.log(2))) / (gamma * np.sqrt(np.pi))) * np.exp(
        -((4 * np.log(2)) / (gamma ** 2)) * (x - mu) ** 2)


def stable_gaussian(x, mu, gamma, a_0):
    """Returns a numerically stable gaussian.
    In the case that non-constant values are found (i.e. not Tiny), then should be normalized to 1.
    Some choices are not clear - like multplying by 4.0 instead of 4.
    We choose not to provide default values because we wish to fail on not passing good values."""
    return a_0 * ((2 * np.sqrt(np.log(2))) / max(tiny, (gamma * np.sqrt(np.pi)))) * np.exp(
        -((4.0 * max(tiny, (np.log(2)) / (gamma ** 2)))) * (
                    x - mu) ** 2)  # here we multiply by 4.0 for the same reason that they multiply by 1.0. Maybe this sets as a float.


def gaussian_derivative(x, mu, gamma, a_0):
    "Returns the normalized first derivative of a gaussian function. Normalized such that the absolute value of the area under the curve is 1"
    return ((- a_0 * 4 * np.log(2) * (x - mu)) / (gamma ** 2)) * np.exp(
        -((4 * np.log(2)) / (gamma ** 2)) * (x - mu) ** 2)
    # The following is the not normalized form. Instead, normalize by multiplying by x?.
    # return ((a_0 * -16 * np.log(2) * np.sqrt(np.log(2)) * (x - mu))/(gamma**3 * np.sqrt(np.pi))) * np.exp(-((4 * np.log(2))/(gamma**2)) * (x - mu)**2 )
