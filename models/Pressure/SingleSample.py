import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft

# Define the fixed length for the magnitude spectrum, matching the model's expected input
# This should be the same as FIXED_SIGNAL_LENGTH in your Model.py (e.g., 6402)
FIXED_SIGNAL_LENGTH = 6402

def generate_normal_signal(
    duration_s: float = 1.0,           # Duration of the signal in seconds
    sampling_rate_hz: int = None,      # Sampling rate in Hz. If None, it's calculated to ensure FIXED_SIGNAL_LENGTH.
    fundamental_freq_hz: float = 60.0, # Frequency of the main cosine wave (e.g., typical vibration/power line freq)
    amplitude: float = 5.0,            # Amplitude of the main cosine wave in the time domain
    noise_amplitude: float = 0.05,     # Amplitude of the epsilon (white) noise to add
    output_filename: str = "normal_observation.txt"
):
    """
    Generates a synthetic 'normal' signal (cosine wave + white noise),
    performs FFT, and saves the Frequency, Magnitude (dB), and Phase (degrees)
    to a tab-separated .txt file.

    The output Magnitude array will be padded/truncated to FIXED_SIGNAL_LENGTH
    to ensure compatibility with your classification model.

    Parameters:
    - duration_s (float): The total duration of the time-domain signal in seconds.
    - sampling_rate_hz (int): The number of samples per second. If None, it's automatically
                              calculated to ensure the FFT's positive frequency spectrum
                              has exactly FIXED_SIGNAL_LENGTH points.
    - fundamental_freq_hz (float): The frequency of the primary cosine wave in Hz.
    - amplitude (float): The peak amplitude of the cosine wave.
    - noise_amplitude (float): The amplitude of the Gaussian (epsilon) noise added.
    - output_filename (str): The name of the .txt file to save the data.
    """
    
    # Calculate the required total number of time-domain samples (n_fft)
    # for the FFT to yield FIXED_SIGNAL_LENGTH positive frequency points.
    n_fft = FIXED_SIGNAL_LENGTH * 2

    # If sampling_rate_hz is not explicitly provided, calculate it
    # to ensure we generate exactly n_fft samples for the given duration.
    if sampling_rate_hz is None:
        sampling_rate_hz = int(n_fft / duration_s)
        # Ensure sampling_rate_hz is at least twice the fundamental frequency for Nyquist
        if sampling_rate_hz < 2 * fundamental_freq_hz:
            print(f"Warning: Calculated sampling rate ({sampling_rate_hz} Hz) is below Nyquist for fundamental frequency ({fundamental_freq_hz} Hz).")
            print(f"Consider increasing duration_s or manually setting sampling_rate_hz >= {2 * fundamental_freq_hz}.")
            # Adjust sampling_rate_hz to meet Nyquist, this might change n_samples slightly if duration is fixed.
            # For this script, we prioritize n_fft, so we'll recalculate sampling_rate_hz to be exactly n_fft/duration_s
            sampling_rate_hz = n_fft / duration_s # Ensure float division for precise calculation
            if not sampling_rate_hz.is_integer():
                print(f"Warning: Sampling rate ({sampling_rate_hz}) is not an integer. Using integer conversion, which might cause small deviations.")
            sampling_rate_hz = int(sampling_rate_hz)

    # Generate time vector with exactly n_fft samples
    t = np.linspace(0, duration_s, n_fft, endpoint=False) # endpoint=False ensures exactly n_fft points

    # Generate the main cosine signal
    raw_signal = amplitude * np.cos(2 * np.pi * fundamental_freq_hz * t)

    # Add epsilon (white) noise from a normal distribution
    noise = noise_amplitude * np.random.randn(n_fft)
    signal_with_noise = raw_signal + noise

    # Perform Fast Fourier Transform (FFT)
    fft_result = fft(signal_with_noise)
    # Calculate corresponding frequencies for the FFT result
    freqs = np.fft.fftfreq(n_fft, 1/sampling_rate_hz)

    # Calculate Magnitude in Decibels (dB)
    # Add a small epsilon to np.abs(fft_result) to avoid log10(0) which results in -inf
    magnitude_db = 20 * np.log10(np.abs(fft_result) + 1e-10)

    # Calculate Phase in Degrees
    phase_deg = np.degrees(np.angle(fft_result))

    # Filter for positive frequencies (excluding the DC component at 0 Hz)
    positive_freq_mask = freqs > 0
    freqs_positive = freqs[positive_freq_mask]
    magnitude_positive = magnitude_db[positive_freq_mask]
    phase_positive = phase_deg[positive_freq_mask]

    # Ensure the number of generated positive frequency points matches FIXED_SIGNAL_LENGTH
    if len(magnitude_positive) > FIXED_SIGNAL_LENGTH:
        # Truncate if more points than required
        freqs_positive = freqs_positive[:FIXED_SIGNAL_LENGTH]
        magnitude_positive = magnitude_positive[:FIXED_SIGNAL_LENGTH]
        phase_positive = phase_positive[:FIXED_SIGNAL_LENGTH]
    elif len(magnitude_positive) < FIXED_SIGNAL_LENGTH:
        # Pad with zeros if fewer points than required (unlikely with even n_fft)
        pad_amount = FIXED_SIGNAL_LENGTH - len(magnitude_positive)
        # Pad frequencies with the last valid frequency or 0 if empty
        freqs_positive = np.pad(freqs_positive, (0, pad_amount), mode='constant', constant_values=freqs_positive[-1] if len(freqs_positive) > 0 else 0)
        magnitude_positive = np.pad(magnitude_positive, (0, pad_amount), mode='constant')
        phase_positive = np.pad(phase_positive, (0, pad_amount), mode='constant')

    # Create a Pandas DataFrame
    df_output = pd.DataFrame({
        "Frequency": freqs_positive,
        "Magnitude": magnitude_positive,
        "Phase": phase_positive
    })

    # Save the DataFrame to a tab-separated .txt file
    df_output.to_csv(output_filename, sep="\t", index=False)

    print(f"Generated signal data saved to '{output_filename}' with {len(df_output)} frequency points.")

    # --- Optional: Plotting the generated spectrum ---
    plt.figure(figsize=(10, 6))
    plt.plot(freqs_positive, magnitude_positive, label="Magnitude (dB)", color='skyblue')
    plt.title(f"Generated Normal Signal Spectrum (Cosine + Noise)\nFile: {output_filename}", fontsize=14, fontweight='bold')
    plt.xlabel("Frequency (Hz)", fontsize=12)
    plt.ylabel("Magnitude (dB)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    # Adjust x-axis limits to clearly show the fundamental frequency peak
    plt.xlim(0, fundamental_freq_hz * 4 + 100) # Show up to 4x fundamental freq + some buffer
    # Adjust y-axis limits to encompass typical magnitude ranges for this type of signal
    plt.ylim(np.min(magnitude_positive) - 10, np.max(magnitude_positive) + 5)
    plt.tight_layout()
    plt.show()

# ========== Example Usage ==========
if __name__ == "__main__":
    print("Generating a sample 'normal' cosine signal with tweakable parameters...")

    # Tweak these parameters to control the generated signal:
    generate_normal_signal(
        duration_s=1.0,           # Signal duration in seconds. Longer duration means finer frequency resolution.
        # sampling_rate_hz=None,  # Leave as None to auto-calculate for FIXED_SIGNAL_LENGTH, or set manually (e.g., 20000)
        fundamental_freq_hz=50.0, # The dominant frequency of the signal (e.g., a machine's operating RPM converted to Hz)
        amplitude=10.0,           # How strong the main signal is
        noise_amplitude=0.1,      # How much random noise is added (smaller = cleaner peak)
        output_filename="normal_test_signal.txt"
    )

    # You can call it again with different parameters to generate other "normal" examples
    # generate_normal_signal(
    #     duration_s=0.5,
    #     fundamental_freq_hz=120.0,
    #     amplitude=8.0,
    #     noise_amplitude=0.08,
    #     output_filename="normal_test_signal_v2.txt"
    # )
