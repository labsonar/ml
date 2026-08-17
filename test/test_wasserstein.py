import numpy as np
from scipy.stats import wasserstein_distance

def calculate_spectrogram_wasserstein(A, B, C, t):
    """
    Calculates the Wasserstein distance between three spectrograms (A, B, C)
    for each frequency bin independently using the real time vector.

    Parameters:
    A, B, C (numpy.ndarray): Input matrices of shape (n_freqs, n_time)
    t (numpy.ndarray): Time vector of shape (n_time,) representing real timestamps

    Returns:
    w_ab, w_ac, w_bc (numpy.ndarray): Vectors of shape (n_freqs,) containing distances in time units
    """
    if not (A.shape == B.shape == C.shape):
        raise ValueError("All three spectrograms must have the exact same shape [freqs, time].")

    if t.shape[0] != A.shape[1]:
        raise ValueError("The length of the time vector must match the number of columns in the spectrograms.")

    n_freqs, n_time = A.shape

    w_ab = np.zeros(n_freqs)
    w_ac = np.zeros(n_freqs)
    w_bc = np.zeros(n_freqs)

    global_min = min(A.min(), B.min(), C.min())
    global_max = max(A.max(), B.max(), C.max())
    denom = global_max - global_min if (global_max - global_min) != 0 else 1.0

    for f in range(n_freqs):
        row_A = A[f, :]
        row_B = B[f, :]
        row_C = C[f, :]

        norm_A = (row_A - global_min) / denom
        norm_B = (row_B - global_min) / denom
        norm_C = (row_C - global_min) / denom

        sum_A = np.sum(norm_A)
        sum_B = np.sum(norm_B)
        sum_C = np.sum(norm_C)

        p_A = norm_A / sum_A if sum_A > 0 else np.ones(n_time) / n_time
        p_B = norm_B / sum_B if sum_B > 0 else np.ones(n_time) / n_time
        p_C = norm_C / sum_C if sum_C > 0 else np.ones(n_time) / n_time

        w_ab[f] = wasserstein_distance(t, t, p_A, p_B)
        w_ac[f] = wasserstein_distance(t, t, p_A, p_C)
        w_bc[f] = wasserstein_distance(t, t, p_B, p_C)

    return w_ab, w_ac, w_bc

import matplotlib.pyplot as plt

# if __name__ == "__main__":
#     np.random.seed(42)
#     freqs = 50
#     tempo = 100

#     t = np.linspace(0, 10, tempo)
#     f_axis = np.linspace(0, 500, freqs)

#     # --- Generating structured data ---
#     # Signals A and B will decay over time (energy concentrated at the beginning)
#     decay_trend = np.exp(-t / 3.0)

#     # Signal C will grow over time (energy concentrated at the end)
#     growth_trend = np.exp((t - 10) / 3.0)

#     # We broadcast these trends across all frequency bins and add a bit of noise
#     sinal_A = np.tile(decay_trend, (freqs, 1)) + np.random.rand(freqs, tempo) * 0.1
#     sinal_B = np.tile(decay_trend, (freqs, 1)) + np.random.rand(freqs, tempo) * 0.1
#     sinal_C = np.tile(growth_trend, (freqs, 1)) + np.random.rand(freqs, tempo) * 0.1

#     # --- Computing the distances ---
#     dist_AB, dist_AC, dist_BC = calculate_spectrogram_wasserstein(sinal_A, sinal_B, sinal_C, t)

#     # --- Plotting the results ---
#     plt.figure(figsize=(10, 6))
#     plt.plot(f_axis, dist_AB, label='A - B (Similar)', color='royalblue', linewidth=2.5)
#     plt.plot(f_axis, dist_AC, label='A - C (Different)', color='darkorange', linewidth=2)
#     plt.plot(f_axis, dist_BC, label='B - C (Different)', color='forestgreen', linewidth=2)

#     plt.title('Wasserstein Distance: Similarity Validation', fontsize=14, fontweight='bold')
#     plt.xlabel('Frequency (Hz)', fontsize=12)
#     plt.ylabel('Wasserstein Distance (seconds)', fontsize=12)

#     # Setting the Y-axis to start at 0 to clearly show the contrast
#     plt.ylim(bottom=0)

#     plt.grid(True, linestyle='--', alpha=0.6)
#     plt.legend(fontsize=11)

#     plt.tight_layout()
#     plt.show()

import sys
import argparse
import lps_sp.acoustical.analysis as lps_analysis
import scipy.io.wavfile as wav

def main():
    parser = argparse.ArgumentParser(
        description="Compute Wasserstein distance between three WAV file spectrograms."
    )

    # Input audio file arguments
    parser.add_argument("wav_a", type=str, help="Path to first WAV file (Signal A)")
    parser.add_argument("wav_b", type=str, help="Path to second WAV file (Signal B)")
    parser.add_argument("wav_c", type=str, help="Path to third WAV file (Signal C)")

    # Inject spectral arguments from your module framework
    lps_analysis.SpectralAnalysis.add_args(parser)

    args = parser.parse_args()

    # Read the 3 WAV files using scipy
    fs_a, data_a = wav.read(args.wav_a)
    fs_b, data_b = wav.read(args.wav_b)
    fs_c, data_c = wav.read(args.wav_c)

    # Ensure they have identical sample rates
    if not (fs_a == fs_b == fs_c):
        raise ValueError("All input WAV files must have the exact same sampling rate.")

    # Initialize analysis parameter class from arguments
    analysis_type, params = lps_analysis.SpectralAnalysis.build_from_args(args)
    print("analysis_type: ", analysis_type)

    # Apply your module's spectral transformation (Spectrogram, LOFAR, or Melgram)
    power_A, freqs_A, times_A = analysis_type.apply(data_a, fs_a, params)
    power_B, _, _ = analysis_type.apply(data_b, fs_b, params)
    power_C, _, _ = analysis_type.apply(data_c, fs_c, params)

    times_A = np.array(times_A)

    # Check shape alignment requirements
    if not (power_A.shape == power_B.shape == power_C.shape):
        print("Error: The calculated spectral matrices have different shapes.", file=sys.stderr)
        print(f"Shapes: A={power_A.shape}, B={power_B.shape}, C={power_C.shape}", file=sys.stderr)
        print("Ensure input files have identical lengths and the same processing parameters.", file=sys.stderr)
        sys.exit(1)

    # Calculate metric distributions across frequencies
    # times_vector = np.array(times_A)
    dist_AB, dist_AC, dist_BC = calculate_spectrogram_wasserstein(power_A, power_B, power_C, times_A)

    mean_spec_A = np.mean(power_A, axis=1)
    mean_spec_B = np.mean(power_B, axis=1)
    mean_spec_C = np.mean(power_C, axis=1)

    # Create a figure with 2 subplots sharing the same X-axis (Frequencies)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    # --- Top Plot: Wasserstein Distance ---
    ax1.plot(freqs_A, dist_AB, label='A - B', color='royalblue', linewidth=2)
    ax1.plot(freqs_A, dist_AC, label='A - C', color='darkorange', linewidth=2)
    ax1.plot(freqs_A, dist_BC, label='B - C', color='forestgreen', linewidth=2)
    ax1.set_title(f'Wasserstein Distance per Frequency Bin ({analysis_type.name})', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Wasserstein Distance (seconds)', fontsize=11)
    ax1.ylim(bottom=0) if hasattr(ax1, 'ylim') else ax1.set_ylim(bottom=0)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(fontsize=10)

    # --- Bottom Plot: Average STFT ---
    ax2.plot(freqs_A, mean_spec_A, label='Signal A', color='crimson', linewidth=1.8)
    ax2.plot(freqs_A, mean_spec_B, label='Signal B', color='teal', linewidth=1.8, linestyle='--')
    ax2.plot(freqs_A, mean_spec_C, label='Signal C', color='darkorchid', linewidth=1.8)

    # Adjust Y-label depending on whether log scale (dB) was used in the module
    y_label = 'Average Power (dB)' if params.log_scale else 'Average Power / Magnitude'
    ax2.set_title('Average STFT Spectrum Over Time', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Frequency (Hz)', fontsize=11)
    ax2.set_ylabel(y_label, fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(fontsize=10)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()