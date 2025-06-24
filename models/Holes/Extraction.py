import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft

# Define the target number of positive frequency points
# This MUST match the FIXED_SIGNAL_LENGTH from your Model.py (which is 6402)
TARGET_POSITIVE_FREQ_POINTS = 6402

def generate_observation_spectrum():
    """
    Generates a single, smooth frequency spectrum for 'Observation.txt'
    with characteristics designed for classification.
    """

    max_freq_hz = 2000 # Your goal plot goes up to 2000 Hz
    freqs_positive = np.linspace(0, max_freq_hz, TARGET_POSITIVE_FREQ_POINTS, endpoint=False)

    # Initialize magnitude spectrum with a very low, smooth baseline
    # Adjusted baseline for smoother appearance
    magnitude_db = np.full(TARGET_POSITIVE_FREQ_POINTS, -40.0, dtype=np.float32)

    # Define peak parameters for this generic observation.
    # These are designed to be somewhat between 'Healthy' and 'Medium Damage'
    # to provide a non-trivial classification target.
    # (Center Frequency (Hz), Peak Magnitude (dB), Bandwidth/Spread (Hz) - increased for smoothness)
    peaks_definition = [
        (105, 32, 40),    # Broader peak than before
        (310, 27, 45),    # Broader peak
        (705, 38, 50),    # Very broad and prominent
        (1105, 35, 55),   # Broader
        (1505, 28, 60),   # Broader
        (1905, 30, 65),   # Broader
        (50, 15, 30)      # A new, relatively minor but broad low-freq component
    ]

    # Add Gaussian peaks to the magnitude spectrum
    for freq_center, peak_magnitude_db, spread_hz in peaks_definition:
        sigma = spread_hz / 3.5 # Slightly increased denominator for even broader peaks

        gaussian_peak = peak_magnitude_db * np.exp(-0.5 * ((freqs_positive - freq_center) / sigma)**2)
        magnitude_db = np.maximum(magnitude_db, gaussian_peak)

    # Add very subtle random fluctuations for a natural look, significantly reduced
    magnitude_db += np.random.normal(0, 0.2, TARGET_POSITIVE_FREQ_POINTS) # +/- 0.2 dB variation

    # Apply a gentle roll-off at higher frequencies, common in real-world systems
    # This creates a slight downward slope towards the right of the spectrum.
    roll_off_factor = np.linspace(1.0, 0.8, TARGET_POSITIVE_FREQ_POINTS) # Gradual decrease
    magnitude_db = magnitude_db * roll_off_factor


    # Generate dummy phase data (model only uses magnitude)
    phase_positive = np.random.uniform(-180, 180, TARGET_POSITIVE_FREQ_POINTS)

    # Save results to file
    df = pd.DataFrame({
        "Frequency": freqs_positive,
        "Magnitude": magnitude_db,
        "Phase": phase_positive
    })
    filename = "Observation.txt"
    df.to_csv(filename, sep="\t", index=False)
    print(f"Analysis complete. Data saved to '{filename}' with {len(df)} frequency points.")

    # Create visualization
    plt.figure(figsize=(14, 7))
    plt.plot(freqs_positive, magnitude_db, 'b-', linewidth=2, label="Magnitude (dB)")
    plt.title(f"Frequency Spectrum Analysis: Simulated Observation", fontsize=16, fontweight='bold')
    plt.xlabel("Frequency (Hz)", fontsize=12)
    plt.ylabel("Magnitude (dB)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.xlim(0, max_freq_hz)
    plt.ylim(-40, 40) # Adjusted for the new baseline and peak amplitudes
    plt.tight_layout()
    plt.show()

    return df

if __name__ == "__main__":
    spectrum_data = generate_observation_spectrum()