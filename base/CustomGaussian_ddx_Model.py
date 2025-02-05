import numpy as np
from lmfit import Model

from .constants import MAX_SIGMA
from .gaussians import stable_gaussian_derivative_sigma


class CustomGaussian_ddx_Model(Model):
    """A custom model of the Gaussian 1st derivative lineshape copying the default lmfit GaussianModel behavior."""

    fwhm_factor = 2 * np.sqrt(2 * np.log(2))
    height_factor = 1 / np.sqrt(2 * np.pi)

    def __init__(self, independent_vars=['x'], prefix='', nan_policy='raise', **kwargs):
        """
        Initialize the CustomGaussianModel.

        Parameters:
        - independent_vars (list): List of independent variable names.
        - prefix (str): Prefix for parameter names.
        - nan_policy (str): Handling of NaN values.
        """
        super().__init__(stable_gaussian_derivative_sigma, prefix=prefix, independent_vars=independent_vars,
                         nan_policy=nan_policy, **kwargs)
        self._set_paramhints_prefix()

    def _set_paramhints_prefix(self):
        """Set parameter hints for amplitude, center, sigma, fwhm, and height."""
        self.set_param_hint(
            'amplitude')  # Allow negative for inverted a term?  Amplitude is probably stritly positive but we might be able to allow negative values for optimization reasons.
        self.set_param_hint('center')
        self.set_param_hint('sigma',min=0, max=MAX_SIGMA)  # Sigma must be positive - can remove to allow the optimizer to range over greater space easily?

        # Expressions for derived parameters fwhm and height
        self.set_param_hint('fwhm', expr=f'{self.prefix}sigma * {self.fwhm_factor}')
        # self.set_param_hint('height', expr=f'{self.prefix}amplitude / ({self.prefix}sigma * {self.height_factor})') double check.

    def guess(self, data, x, negative=False, **kwargs):
        """
        Estimate initial parameter values from data.
        This is just code copying the default behavior from lmfit.
        We should never use this code because we have a sophisticated initial guess algo.

        Parameters:
        - data (array): The dependent data.
        - x (array): The independent variable.
        - negative (bool): Whether to invert the data for peak finding.

        Returns:
        - Parameters: An lmfit.Parameters object with initial guesses.
        """
        # Guess the peak amplitude
        amplitude_guess = np.max(data) if not negative else np.min(data)
        # Guess the peak center
        center_guess = x[np.argmax(data)] if not negative else x[np.argmin(data)]
        # Guess sigma as a fraction of the total range
        sigma_guess = (x.max() - x.min()) / 6.0  # Rough estimate for sigma

        params = self.make_params(amplitude=amplitude_guess, center=center_guess, sigma=sigma_guess)
        return params


