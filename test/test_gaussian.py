import argparse
import sys
import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from scipy.stats import normaltest
import lps_sp.acoustical.analysis as lps_analysis

def analyze_frequency_gaussianity(power_matrix):
    """
    Performs a normality test for each frequency bin over time.
    """
    n_freqs, n_time = power_matrix.shape
    p_values = np.zeros(n_freqs)

    if n_time < 8:
        raise ValueError("Not enough time samples to perform a reliable normality test (minimum 8).")

    for f in range(n_freqs):
        time_series = power_matrix[f, :]

        if np.all(time_series == time_series[0]):
            p_values[f] = 0.0
            continue

        _, p_val = normaltest(time_series)
        p_values[f] = p_val

    return p_values

def main():
    parser = argparse.ArgumentParser(
        description="Prospection: Analyze Gaussianity with p-values and target histograms."
    )

    parser.add_argument("wav_file", type=str, help="Path to the WAV file to analyze")

    lps_analysis.SpectralAnalysis.add_args(parser)
    args = parser.parse_args()

    fs, data = wav.read(args.wav_file)
    data = data.astype(np.float32) / 32768.0

    analysis_type, params = lps_analysis.SpectralAnalysis.build_from_args(args)

    power, freqs, times = analysis_type.apply(data, fs, params)

    p_values = analyze_frequency_gaussianity(power)
    mean_spec = np.mean(power, axis=1)

    gaussian_percentage = (np.sum(p_values > 0.05) / len(freqs)) * 100

    # --- Target Frequencies for Histograms ---
    target_freqs = [1000.0, 5000.0, 7000.0]
    target_indices = []

    # Find the closest frequency bin available for each target
    for tf in target_freqs:
        idx = np.argmin(np.abs(freqs - tf))
        target_indices.append(idx)

    # --- Figure Setup (Grid Layout) ---
    # Top rows will span across columns for the full spectrum plots
    fig = plt.figure(figsize=(12, 12))
    grid = plt.GridSpec(3, 3, wspace=0.3, hspace=0.4)

    ax1 = fig.add_subplot(grid[0, :])  # Top plot: p-value
    ax2 = fig.add_subplot(grid[1, :])  # Middle plot: average STFT

    # Bottom row: Histograms for the 3 target frequencies
    ax_hist1 = fig.add_subplot(grid[2, 0])
    ax_hist2 = fig.add_subplot(grid[2, 1])
    ax_hist3 = fig.add_subplot(grid[2, 2])
    hist_axes = [ax_hist1, ax_hist2, ax_hist3]
    colors = ['crimson', 'teal', 'darkorange']

    # --- 1. Top Plot: Normality Test (p-value) ---
    ax1.plot(freqs, p_values, color='darkorchid', linewidth=1.8, label='p-value')
    ax1.axhline(y=0.05, color='crimson', linestyle='--', linewidth=1.5, label='Alpha Significance (0.05)')
    ax1.fill_between(freqs, p_values, 0.05, where=(p_values > 0.05), color='green', alpha=0.1, label='Gaussian Bins')

    # Mark where the target frequencies are on the p-value curve
    for idx, tf in zip(target_indices, target_freqs):
        ax1.plot(freqs[idx], p_values[idx], 'o', markersize=8, markeredgecolor='black', label=f'Bin {freqs[idx]:.0f} Hz')

    ax1.set_title(f"Gaussianity Test per Frequency Bin ({analysis_type.name})\n{gaussian_percentage:.1f}% of bins are sufficiently Gaussian",
                  fontsize=12, fontweight='bold')
    ax1.set_ylabel('p-value', fontsize=11)
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(fontsize=9, loc='upper right')

    # --- 2. Middle Plot: Average STFT ---
    ax2.plot(freqs, mean_spec, color='black', linewidth=1.5, label='Average Spectrum')
    y_label = 'Average Power (dB)' if params.log_scale else 'Average Power / Magnitude'
    ax2.set_title('Average STFT Spectrum Over Time', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Frequency (Hz)', fontsize=11)
    ax2.set_ylabel(y_label, fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.6)

    # --- 3. Bottom Plots: Target Histograms ---
    for ax, idx, color_h in zip(hist_axes, target_indices, colors):
        actual_f = freqs[idx]
        p_val_f = p_values[idx]
        time_series_data = power[idx, :]

        # Plot the histogram of amplitudes over time
        ax.hist(time_series_data, bins=80, color=color_h, edgecolor='black', alpha=0.7, density=True)

        # Define status based on p-value
        status = "Gaussian" if p_val_f > 0.05 else "Non-Gaussian"

        ax.set_title(f"Freq: {actual_f:.0f} Hz\np-val: {p_val_f:.4f} ({status})", fontsize=10, fontweight='bold')
        ax.set_xlabel('Value (Amplitude/dB)', fontsize=9)
        ax.set_ylabel('Density', fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.5)

    plt.show()

if __name__ == "__main__":
    main()