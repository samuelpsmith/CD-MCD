# Test data generation.
import numpy as np

NUM_GAUSSIANS = 3  # Number of Gaussians to generate
NOISE_LEVEL = 0.01  # Noise level to add to the generated Gaussian curve
NUM_X_VALUES = 500  # Number of points in the x-axis array
X_MIN = 0  # Minimum value for the x range
X_MAX = 60  # Maximum value for the x range
SEED = 479806  # seed for reproducibility.
MIN_DISTANCE = 5  # simulate the 'bandwidth' of the fake instrument.

#Smoothing
WINDOW_LENGTH = 5  # Window length for Savitzky-Golay smoothing (datapoints?) (relate to bandwidth?)
POLYORDER = 4  # Polynomial order for Savitzky-Golay smoothing

#peak picking
HEIGHT_THRESHOLD = 0.04  # Minimum height threshold for peak detection - should be greater than noise after smoothing. Trouble is that the negative side bands likely mean zero isn't at zero.
DISTANCE = 5  # The minimum distance, in number of samples, between peaks. - This should be related to bandwidth for certain
PROMINENCE_PERECENT = 0.04  # Prominence is here as a multiple of max height. What is (topographic) prominence? It is "the minimum height necessary to descend to get from the summit to any higher terrain", as it can be seen here

# Fitting
MAX_BASIS_GAUSSIANS = 10  # Maximum number of basis Gaussians for fitting
NUM_GUESSES = 1  # Number of guesses for each basis fitting - created this with the idea that I would want to allow for some random walk over guess space ? For now keep to 1.
VARY_CENTERS = False
DELTA_BIC_THRESHOLD = 0  # Threshold for detecting when BIC levels out - depends on the number of points. Need to normalize somehow. Set to zero to effectively bypass.
THRESHOLD_PERCENT = 0.1  # Threshold under which a basis curve is not contributing to the fit, and therefore is removed. This should probably be maximally 100/N, N is the number of (reasonable) basis curves. In experience, the ones that don't contribute tend to be very small say less than 1%
PERCENTAGE_RANGE = 1  # The percentage by which the initial parameters will be allowed to relax on re-fitting after removing poor curves.
PERCENT_RANGE_X = 1
TOLERANCE_X = 2  # how close centers can be as a percent of the overall x values and be considered the same.
MAX_SIGMA = 60000 #max sigma for gaussians
MIN_PEAK_X_DISTANCE = 0
ESTIMATE_SIGMA_ITERATIONS = 10 #1/n max 2/n max, 3/n max ... (n-1)/n.. like FWHM

#basic
SMALL_FWHM_FACTOR = 2.355  # Conversion factor from FWHM to sigma
tiny = 1.0e-15
log2 = np.log(2)
s2pi = np.sqrt(2 * np.pi)
s2 = np.sqrt(2.0)
