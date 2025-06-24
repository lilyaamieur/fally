import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft # Although we're generating in freq domain, keeping for consistency if time domain needed later
from tqdm import tqdm # For progress bar

# ============ GLOBAL CONFIGURATION ============
# This should match the FIXED_SIGNAL_LENGTH in your Model.py
FIXED_SIGNAL_LENGTH = 6402
MAX_FREQ_HZ = 2000 # Max frequency to simulate, matching your EDA plots

# ============ PARAMETER TWEAKING SECTION ============
# You can adjust these parameters to define the characteristics of each stress level.
# The parameters are:
#   - 'base_magnitude_db': The baseline magnitude (dB) for the spectrum.
#   - 'overall_gain': Multiplier for the final magnitude (linear scale).
#   - 'noise_amplitude_db': Standard deviation (dB) for random fluctuations on top of the spectrum.
#   - 'peaks': A list of (center_freq_hz, peak_magnitude_db, bandwidth_hz) tuples for Gaussian peaks.
#              - center_freq_hz: The frequency where the peak is highest.
#              - peak_magnitude_db: The maximum amplitude of the peak in dB.
#              - bandwidth_hz: Controls the 'width' or 'Q-factor' of the peak. Larger = wider/smoother.

STRESS_PARAMETERS = {
    "healthy": {
        'base_magnitude_db': -80.0, # Much lower baseline for very clean signals
        'overall_gain': 1.0,
        'noise_amplitude_db': 0.1, # Extremely low noise for healthy
        'peaks': [
            (100, 40, 5),    # Very sharp peaks
            (250, 30, 6),
            (400, 45, 7),
            (600, 35, 8),
            (850, 42, 9),
            (1100, 38, 10),
            (1350, 40, 11),
            (1600, 37, 12),
            (1850, 39, 13),
        ]
    },
    "light stress": {
        'base_magnitude_db': -70.0, # Slightly higher baseline than healthy
        'overall_gain': 0.98,
        'noise_amplitude_db': 0.3, # Still very low noise, but slightly more than healthy
        'peaks': [
            (102, 38, 7),    # Slight shift, slightly wider
            (255, 28, 8),
            (405, 43, 9),
            (605, 33, 10),
            (855, 40, 11),
            (1105, 36, 12),
            (1355, 38, 13),
            (1605, 35, 14),
            (1855, 37, 15),
            (50, 10, 10),      # Introduction of a minor new, sharp peak
        ]
    },
    "medium stress": {
        'base_magnitude_db': -60.0, # Noticeably higher baseline
        'overall_gain': 0.95,
        'noise_amplitude_db': 0.7, # More noticeable, but still controlled noise
        'peaks': [
            (105, 35, 10),    # More significant shift, wider
            (260, 25, 11),
            (410, 40, 12),
            (610, 30, 13),
            (860, 37, 14),
            (1110, 33, 15),
            (1360, 35, 16),
            (1610, 32, 17),
            (1860, 34, 18),
            (55, 15, 12),      # More prominent new peak, still sharp
            (200, 8, 15),      # Another new, smaller peak
        ]
    },
    "heavy stress": {
        'base_magnitude_db': -50.0, # Highest baseline for heavy stress
        'overall_gain': 0.9,
        'noise_amplitude_db': 1.2, # Highest noise, but not excessively spiky
        'peaks': [
            (110, 30, 15),    # Significant shift, wider
            (265, 20, 16),
            (415, 35, 17),
            (615, 25, 18),
            (865, 32, 19),
            (1115, 28, 20),
            (1365, 30, 21),
            (1615, 27, 22),
            (1865, 29, 23),
            (60, 20, 18),      # Very prominent new low-freq peak
            (220, 15, 20),     # Clear new peak
            (1000, 10, 25)     # A broader, general increase or new resonance
        ]
    }
}


# ============ SIGNAL GENERATION FUNCTION ============
def generate_stress_signal(
    stress_level: str,
    output_filename: str
) -> pd.DataFrame:
    """
    Generates a synthetic frequency spectrum based on a specified stress level,
    and saves the Frequency, Magnitude (dB), and Phase (degrees) to a .txt file.

    The output Magnitude array is guaranteed to have FIXED_SIGNAL_LENGTH points.

    Parameters:
    - stress_level (str): Must be one of the keys in STRESS_PARAMETERS (e.g., "healthy", "light stress").
    - output_filename (str): The name of the .txt file to save the data.

    Returns:
    - pd.DataFrame: The generated data.
    """

    if stress_level not in STRESS_PARAMETERS:
        raise ValueError(f"Invalid stress_level: '{stress_level}'. Must be one of {list(STRESS_PARAMETERS.keys())}")

    params = STRESS_PARAMETERS[stress_level]

    # Create the frequency array, ensuring it has FIXED_SIGNAL_LENGTH points
    freqs_positive = np.linspace(0, MAX_FREQ_HZ, FIXED_SIGNAL_LENGTH, endpoint=False)

    # Initialize magnitude spectrum with the defined baseline
    magnitude_db = np.full(FIXED_SIGNAL_LENGTH, params['base_magnitude_db'], dtype=np.float32)

    # Add Gaussian peaks
    for freq_center, peak_magnitude_db, bandwidth_hz in params['peaks']:
        # sigma for Gaussian is related to bandwidth (standard deviation)
        # Higher bandwidth_hz means a wider peak, mimicking more damping/broader resonance
        # Reduced denominator for a sharper Gaussian peak
        sigma = bandwidth_hz / 1.5 # Adjusted for much sharper peaks

        gaussian_peak = peak_magnitude_db * np.exp(-0.5 * ((freqs_positive - freq_center) / sigma)**2)
        magnitude_db = np.maximum(magnitude_db, gaussian_peak) # Add peaks on top of baseline

    # Add random fluctuations (noise)
    magnitude_db += np.random.normal(0, params['noise_amplitude_db'], FIXED_SIGNAL_LENGTH)

    # Apply overall gain/level adjustment
    magnitude_db *= params['overall_gain']

    # Generate dummy phase data (often random or loosely correlated in such analyses)
    phase_positive = np.random.uniform(-180, 180, FIXED_SIGNAL_LENGTH)

    # Create DataFrame and save
    df_output = pd.DataFrame({
        "Frequency": freqs_positive,
        "Magnitude": magnitude_db,
        "Phase": phase_positive
    })
    df_output.to_csv(output_filename, sep="\t", index=False)

    return df_output

# ============ DATASET GENERATION SCRIPT ============
def generate_stress_dataset(
    base_output_dir: str = "./synthetic_stress_data",
    num_files_per_folder: int = 50, # N number of files per folder
    plot_sample_for_each_type: bool = True # Set to True to see an example plot for each generated type
):
    """
    Creates folders and generates synthetic signal data files for different stress levels.

    Parameters:
    - base_output_dir (str): The root directory where the stress level folders will be created.
    - num_files_per_folder (int): The number of .txt files to generate in each stress level folder.
    - plot_sample_for_each_type (bool): If True, plots one example spectrum for each stress level.
    """

    print(f"Starting synthetic dataset generation in: {os.path.abspath(base_output_dir)}")
    os.makedirs(base_output_dir, exist_ok=True)

    generated_file_paths = []

    for stress_level in STRESS_PARAMETERS.keys():
        folder_path = os.path.join(base_output_dir, stress_level.replace(" ", "_"))
        os.makedirs(folder_path, exist_ok=True)
        print(f"\nCreating files for: '{stress_level}' in '{folder_path}'")

        # Keep track of one file path for plotting later
        sample_df_to_plot = None

        for i in tqdm(range(num_files_per_folder), desc=f"Generating {stress_level} files"):
            filename = f"{stress_level.replace(' ', '_')}_{i+1:04d}.txt"
            filepath = os.path.join(folder_path, filename)
            
            # Generate the signal data
            df = generate_stress_signal(stress_level, filepath)
            generated_file_paths.append(filepath)

            if i == 0 and plot_sample_for_each_type: # Keep the first generated df for plotting
                sample_df_to_plot = df

        if plot_sample_for_each_type and sample_df_to_plot is not None:
            # Plot the first sample generated for this stress level
            plt.figure(figsize=(12, 6))
            plt.plot(sample_df_to_plot["Frequency"], sample_df_to_plot["Magnitude"], 'b-', linewidth=1.5)
            plt.title(f"Sample Spectrum: '{stress_level}'", fontsize=14, fontweight='bold')
            plt.xlabel("Frequency (Hz)", fontsize=10)
            plt.ylabel("Magnitude (dB)", fontsize=10)
            plt.grid(True, alpha=0.3)
            # Adjust x and y limits for the new type of signal
            plt.xlim(0, MAX_FREQ_HZ)
            # Y-axis will need to be adjusted significantly lower
            plt.ylim(STRESS_PARAMETERS[stress_level]['base_magnitude_db'] - 5,
                     np.max([p[1] for p in STRESS_PARAMETERS[stress_level]['peaks']]) + 5)
            plt.tight_layout()
            plt.show()

    print(f"\nDataset generation complete. Total files generated: {len(generated_file_paths)}")
    return generated_file_paths

# ============ MAIN EXECUTION ============
if __name__ == "__main__":
    # --- TWEAKABLE EXECUTION PARAMETERS ---
    output_directory = "./my_stress_data" # Change this to your desired output folder
    num_samples = 100                    # Number of files to generate in EACH folder (e.g., 100 for healthy, 100 for light stress, etc.)
    show_example_plots = True           # Set to False if you don't want plots popping up

    # Call the main dataset generation function
    generated_files = generate_stress_dataset(
        base_output_dir=output_directory,
        num_files_per_folder=num_samples,
        plot_sample_for_each_type=show_example_plots
    )

    # You can now use 'generated_files' list for further processing if needed.
    # For example, pass it to your EDA script or your Model.py by adjusting DATA_DIR.
    print(f"\nGenerated dataset available in: {os.path.abspath(output_directory)}")
