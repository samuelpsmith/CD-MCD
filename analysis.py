import numpy as np
import pandas as pd
import tkinter as tk
import tkinter.filedialog
import matplotlib.pyplot as plt
import scipy.fft
from scipy.fft import fft, fftfreq
from scipy.interpolate import interp1d
import scipy.signal.windows as win
from scipy.stats import triang

def generate_overlapping_bands(x, seperation, num_g, seed, start):
    prev_g = np.zeros_like(gaussian(1,1,1,1))
    np.random.seed(seed)
    # Generate random amplitude and width for the Gaussian
    for i in range(0,num_g+1):
        gamma = np.random.uniform(30, 200)
        amp = np.random.uniform(1000,1500)
        g = gaussian(x, start, gamma, amp) + prev_g
        prev_g = g
        start = start + seperation
    return (x,g)
def gaussian(x, mu, gamma, a_0):
    "Returns a normalized gaussian function. Normalized to 1"
    return a_0 * ((2.0 * np.sqrt(np.log(2))) / (gamma * np.sqrt(np.pi))) * np.exp(
        -((4.0 * np.log(2)) / (gamma ** 2.0)) * (x - mu) ** 2.0)
def gaussian2(x, mu, sig, a_0):
    return a_0 * (
        1.0 / (np.sqrt(2.0 * np.pi) * sig) * np.exp(-np.power((x - mu) / sig, 2.0) / 2))
def select_processed_file():
    root = tk.Tk()
    root.withdraw()
    file_path = tk.filedialog.askopenfilename(title="Select processed file")
    return file_path

def main():
    #fp = select_processed_file()
    #df = pd.read_csv(fp)

    #for fft
    #wavelength = df.get("wavelength")
    #abs =  df.get("abs")

    wavelength = np.arange(150,850,2)
    data = generate_overlapping_bands(wavelength, 80, 5, 123, 300)
    abs = data[1]

    #filter nans
    mask = ~np.isnan(wavelength) & ~np.isnan(abs)

    #use mask
    xs = wavelength[mask]
    ys = abs[mask]
    neg_xs = xs * -1
    all_values = list(zip(neg_xs, ys)) + list(zip(xs, ys))
    all_values.sort(key=lambda pair_xy: pair_xy[0])

    xs_all = [pair[0] for pair in all_values]
    ys_all = [pair[1] for pair in all_values]


    plt.plot(xs,ys)
    plt.grid()
    plt.show()
    ig = np.fft.ifft(ys)
    windowed_ig = win.flattop(len(ig),False) * ig
    fft_res = np.fft.fft(windowed_ig)
    plt.plot(xs,ig)
    plt.show()
    plt.plot(xs,windowed_ig)
    plt.show()
    plt.plot(xs,np.abs(fft_res))
    plt.plot()
    plt.show()



if __name__ == "__main__":
    main()