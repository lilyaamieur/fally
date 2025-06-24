import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft
from tqdm import tqdm
import random

# ============ GLOBAL CONFIGURATION ============
FIXED_SIGNAL_LENGTH = 6402
MAX_FREQ_HZ = 2000
SAMPLING_RATE = 2 * MAX_FREQ_HZ  # Nyquist criterion - used for conceptual time-domain conversion

# ============ REALISTIC PARAMETER TWEAKING SECTION ============
# --- MODIFIED: Ranges now overlap significantly to make classes harder to distinguish ---
STRESS_PARAMETERS = {
    "healthy": {
        'primary_modes': [
            {'freq': 45.2, 'damping_range': (0.007, 0.012), 'amplitude_range': (30, 60)},
            {'freq': 127.8, 'damping_range': (0.010, 0.016), 'amplitude_range': (25, 55)},
            {'freq': 251.4, 'damping_range': (0.013, 0.020), 'amplitude_range': (20, 50)},
            {'freq': 412.6, 'damping_range': (0.016, 0.025), 'amplitude_range': (18, 45)},
            {'freq': 634.9, 'damping_range': (0.020, 0.030), 'amplitude_range': (15, 40)},
            {'freq': 892.3, 'damping_range': (0.023, 0.035), 'amplitude_range': (12, 35)},
            {'freq': 1156.7, 'damping_range': (0.026, 0.040), 'amplitude_range': (10, 30)},
            {'freq': 1423.1, 'damping_range': (0.030, 0.045), 'amplitude_range': (8, 28)},
            {'freq': 1689.5, 'damping_range': (0.033, 0.050), 'amplitude_range': (6, 26)},
        ],
        'broadband_noise_db': -70,
        'measurement_noise_std': 0.7,
        'freq_jitter_std': 1.2, # === MODIFIED: Increased jitter
        'amplitude_jitter_factor': 0.20, # === MODIFIED: Increased jitter
        'harmonic_distortion': 0.03,
        'baseline_drift_range': (-2, 2),
        'transient_event_prob': 0.08, 
        'max_transient_peaks_per_sample': 2,
        'transient_amplitude_range_db': (50, 75),
        'transient_bandwidth_range_hz': (1, 6),
    },
    
    "light stress": {
        'primary_modes': [ # Frequencies slightly lower, damping ranges slightly higher but overlapping
            {'freq': 45.0, 'damping_range': (0.009, 0.018), 'amplitude_range': (28, 58)},
            {'freq': 127.5, 'damping_range': (0.013, 0.022), 'amplitude_range': (23, 53)},
            {'freq': 251.0, 'damping_range': (0.016, 0.028), 'amplitude_range': (18, 48)},
            {'freq': 412.2, 'damping_range': (0.020, 0.034), 'amplitude_range': (16, 43)},
            {'freq': 634.4, 'damping_range': (0.024, 0.039), 'amplitude_range': (13, 38)},
            {'freq': 891.8, 'damping_range': (0.027, 0.044), 'amplitude_range': (11, 33)},
            {'freq': 1156.1, 'damping_range': (0.030, 0.049), 'amplitude_range': (9, 28)},
            {'freq': 1422.5, 'damping_range': (0.034, 0.054), 'amplitude_range': (7, 26)},
            {'freq': 1688.9, 'damping_range': (0.037, 0.059), 'amplitude_range': (5, 24)},
        ],
        'broadband_noise_db': -68,
        'measurement_noise_std': 1.0,
        'freq_jitter_std': 1.6, # === MODIFIED: Increased jitter
        'amplitude_jitter_factor': 0.25, # === MODIFIED: Increased jitter
        'harmonic_distortion': 0.045,
        'baseline_drift_range': (-3, 3),
        'transient_event_prob': 0.15,
        'max_transient_peaks_per_sample': 3,
        'transient_amplitude_range_db': (55, 80),
        'transient_bandwidth_range_hz': (1.5, 8),
    },
    
    "medium stress": {
        'primary_modes': [
            {'freq': 44.7, 'damping_range': (0.012, 0.025), 'amplitude_range': (25, 55)},
            {'freq': 127.1, 'damping_range': (0.017, 0.033), 'amplitude_range': (20, 50)},
            {'freq': 250.4, 'damping_range': (0.021, 0.038), 'amplitude_range': (16, 45)},
            {'freq': 411.6, 'damping_range': (0.025, 0.045), 'amplitude_range': (14, 40)},
            {'freq': 633.7, 'damping_range': (0.030, 0.050), 'amplitude_range': (11, 35)},
            {'freq': 891.1, 'damping_range': (0.034, 0.055), 'amplitude_range': (9, 30)},
            {'freq': 1155.3, 'damping_range': (0.038, 0.060), 'amplitude_range': (7, 27)},
            {'freq': 1421.7, 'damping_range': (0.042, 0.065), 'amplitude_range': (5, 25)},
            {'freq': 1688.1, 'damping_range': (0.046, 0.070), 'amplitude_range': (4, 22)},
        ],
        'broadband_noise_db': -65,
        'measurement_noise_std': 1.5,
        'freq_jitter_std': 2.0, # === MODIFIED: Increased jitter
        'amplitude_jitter_factor': 0.30, # === MODIFIED: Increased jitter
        'harmonic_distortion': 0.06,
        'baseline_drift_range': (-4.5, 4.5),
        'transient_event_prob': 0.25,
        'max_transient_peaks_per_sample': 4,
        'transient_amplitude_range_db': (60, 85),
        'transient_bandwidth_range_hz': (2, 10),
    },
    
    "heavy stress": {
        'primary_modes': [
            {'freq': 44.3, 'damping_range': (0.018, 0.035), 'amplitude_range': (20, 50)},
            {'freq': 126.6, 'damping_range': (0.024, 0.042), 'amplitude_range': (17, 47)},
            {'freq': 249.7, 'damping_range': (0.029, 0.050), 'amplitude_range': (14, 44)},
            {'freq': 410.8, 'damping_range': (0.034, 0.058), 'amplitude_range': (11, 41)},
            {'freq': 632.9, 'damping_range': (0.040, 0.065), 'amplitude_range': (9, 38)},
            {'freq': 890.2, 'damping_range': (0.045, 0.072), 'amplitude_range': (7, 34)},
            {'freq': 1154.4, 'damping_range': (0.050, 0.078), 'amplitude_range': (6, 30)},
            {'freq': 1420.8, 'damping_range': (0.055, 0.085), 'amplitude_range': (4, 28)},
            {'freq': 1687.2, 'damping_range': (0.060, 0.090), 'amplitude_range': (3, 25)},
        ],
        'broadband_noise_db': -62,
        'measurement_noise_std': 2.0,
        'freq_jitter_std': 2.5, # === MODIFIED: Increased jitter
        'amplitude_jitter_factor': 0.35, # === MODIFIED: Increased jitter
        'harmonic_distortion': 0.085,
        'baseline_drift_range': (-6, 6),
        'transient_event_prob': 0.40,
        'max_transient_peaks_per_sample': 5,
        'transient_amplitude_range_db': (65, 90),
        'transient_bandwidth_range_hz': (3, 12),
    }
}

# Environmental and operational variability parameters (these apply universally per sample)
ENVIRONMENTAL_EFFECTS = {
    'temperature_sensitivity': 0.002,
    'humidity_effect': 0.0005,
    'wind_loading_effect': 0.001,
    'traffic_loading_prob': 0.15,
}

def generate_realistic_signal(stress_level: str, output_filename: str) -> pd.DataFrame:
    """
    Generate much more realistic structural health monitoring data with:
    - Proper modal parameters from structural dynamics with increased per-sample randomness.
    - Environmental effects
    - Measurement noise and artifacts
    - Operational variability
    - Realistic frequency content
    - **Random, large transient spikes (frequencies and strengths vary)**
    """
    
    if stress_level not in STRESS_PARAMETERS:
        raise ValueError(f"Invalid stress_level: '{stress_level}'. Must be one of {list(STRESS_PARAMETERS.keys())}")

    params = STRESS_PARAMETERS[stress_level]
    freqs = np.linspace(0, MAX_FREQ_HZ, FIXED_SIGNAL_LENGTH, endpoint=False)
    
    # Start with realistic broadband noise floor
    magnitude_db = np.full(FIXED_SIGNAL_LENGTH, params['broadband_noise_db'], dtype=np.float64)
    
    # Add 1/f noise (pink noise) - common in structural systems
    pink_noise_power = 1.0 / (freqs + 1e-10) # Add epsilon to avoid division by zero at freqs[0]
    pink_noise_db = 10 * np.log10(pink_noise_power + 1e-10)
    pink_noise_db = (pink_noise_db - np.max(pink_noise_db)) * 8 + params['broadband_noise_db']
    magnitude_db = np.maximum(magnitude_db, pink_noise_db * 0.3) # Combine with broadband noise

    # Environmental effects (simulate temperature, humidity variations) - applied universally
    temp_variation = np.random.normal(0, 10)
    humidity_variation = np.random.normal(0, 20)
    wind_effect = np.random.normal(0, 5)
    
    # +++ NEW FEATURE: Global Excitation Factor +++
    # This simulates the overall energy of the input (e.g., strong vs. weak waves/wind)
    # It's a major source of ambiguity between classes.
    global_excitation_factor = np.random.uniform(0.7, 1.3)
    
    # Add structural modes with realistic modal parameters and increased randomness
    for mode in params['primary_modes']:
        base_freq = mode['freq']
        
        # Environmental frequency shifts
        freq_shift = (temp_variation * ENVIRONMENTAL_EFFECTS['temperature_sensitivity'] + 
                      humidity_variation * ENVIRONMENTAL_EFFECTS['humidity_effect'] +
                      wind_effect * ENVIRONMENTAL_EFFECTS['wind_loading_effect'])
        
        # Add measurement-to-measurement random frequency jitter
        freq_jitter = np.random.normal(0, params['freq_jitter_std'])
        actual_freq = base_freq * (1 + freq_shift) + freq_jitter
        
        # Random amplitude variation due to operational conditions
        base_amplitude = np.random.uniform(*mode['amplitude_range'])
        amplitude_jitter = 1 + np.random.normal(0, params['amplitude_jitter_factor'])
        # === MODIFIED: Apply the global excitation factor ===
        actual_amplitude = base_amplitude * amplitude_jitter * global_excitation_factor
        
        # Random damping variation from a range
        # === MODIFIED: Damping is now sampled from a range, not jittered from a fixed point ===
        actual_damping_ratio = np.random.uniform(*mode['damping_range'])
        actual_damping_ratio = np.clip(actual_damping_ratio, 0.001, 0.15) # Ensure damping is physical
        
        # Create realistic modal peak using proper damping (Lorentzian)
        Q_factor = 1 / (2 * actual_damping_ratio)
        bandwidth = actual_freq / Q_factor
        
        lorentzian = actual_amplitude / (1 + ((freqs - actual_freq) / (bandwidth/2))**2)
        
        # Convert to dB and add to spectrum, maintaining a high baseline for peaks
        lorentzian_db = 20 * np.log10(lorentzian + 1e-10)
        magnitude_db = np.maximum(magnitude_db, 
                                  lorentzian_db + params['broadband_noise_db'] + 60) # Offset to make peaks stand out
    
    # Add harmonics and intermodulation products (realistic nonlinear effects)
    if params['harmonic_distortion'] > 0:
        for mode in params['primary_modes'][:3]:
            if np.random.random() < 0.3:
                harmonic_freq = mode['freq'] * 2
                if harmonic_freq < MAX_FREQ_HZ:
                    # === MODIFIED: Harmonic amplitude is also affected by excitation ===
                    harmonic_amp = np.random.uniform(5, 15) * params['harmonic_distortion'] * global_excitation_factor
                    sigma_harmonic = np.random.uniform(2, 8) # Random width
                    harmonic_gaussian = harmonic_amp * np.exp(-0.5 * ((freqs - harmonic_freq) / sigma_harmonic)**2)
                    magnitude_db = np.maximum(magnitude_db, harmonic_gaussian)
    
    # Traffic loading effects (if applicable)
    if np.random.random() < ENVIRONMENTAL_EFFECTS['traffic_loading_prob']:
        traffic_freqs = np.random.uniform(1, 8, size=np.random.randint(2, 5))
        for tf in traffic_freqs:
            traffic_amp = np.random.uniform(8, 18)
            traffic_sigma = np.random.uniform(0.5, 3.0)
            traffic_gaussian = traffic_amp * np.exp(-0.5 * ((freqs - tf) / traffic_sigma)**2)
            magnitude_db = np.maximum(magnitude_db, traffic_gaussian)
    
    # === NEW: Add truly random, large transient spikes ===
    if np.random.random() < params['transient_event_prob']:
        num_transient_peaks = np.random.randint(1, params['max_transient_peaks_per_sample'] + 1)
        for _ in range(num_transient_peaks):
            transient_freq = np.random.uniform(20, MAX_FREQ_HZ - 20) # Random frequency
            transient_amp = np.random.uniform(*params['transient_amplitude_range_db']) # Random high amplitude
            transient_bandwidth = np.random.uniform(*params['transient_bandwidth_range_hz']) # Random sharpness
            
            sigma_transient = transient_bandwidth / 1.5 # Convert bandwidth to Gaussian sigma
            
            transient_gaussian = transient_amp * np.exp(-0.5 * ((freqs - transient_freq) / sigma_transient)**2)
            magnitude_db = np.maximum(magnitude_db, transient_gaussian) # Add on top
    
    # Add measurement noise (includes ADC quantization, sensor noise, etc.)
    measurement_noise = np.random.normal(0, params['measurement_noise_std'], FIXED_SIGNAL_LENGTH)
    magnitude_db += measurement_noise
    
    # Add baseline drift (common in real measurements)
    baseline_drift = np.random.uniform(*params['baseline_drift_range'])
    magnitude_db += baseline_drift
    
    # Add subtle frequency-dependent gain variations (instrument response)
    freq_response_variation = 0.5 * np.sin(2 * np.pi * freqs / 1000) * np.random.uniform(0.5, 1.5)
    magnitude_db += freq_response_variation
    
    # Simulate realistic phase behavior
    phase_degrees = np.zeros(FIXED_SIGNAL_LENGTH)
    for i, mode in enumerate(params['primary_modes']):
        mode_freq = mode['freq']
        # === MODIFIED: Use the actual damping value for this sample for phase calculation ===
        damping_for_phase = actual_damping_ratio # This is an approximation, but better than using a fixed value
        
        freq_ratio = freqs / (mode_freq + 1e-10) # Avoid div by zero
        mode_phase = -np.arctan2(2 * damping_for_phase * freq_ratio, 
                                1 - freq_ratio**2) * 180 / np.pi
        
        weight = np.exp(-((freqs - mode_freq) / (mode_freq * 0.1))**2)
        phase_degrees += mode_phase * weight
    
    phase_noise = np.random.normal(0, 10, FIXED_SIGNAL_LENGTH) # Increased phase noise
    phase_degrees += phase_noise
    
    phase_degrees = ((phase_degrees + 180) % 360) - 180
    
    # Create DataFrame
    df_output = pd.DataFrame({
        "Frequency": freqs,
        "Magnitude": magnitude_db,
        "Phase": phase_degrees
    })
    
    df_output.to_csv(output_filename, sep="\t", index=False)
    return df_output

# The rest of the generation/plotting script remains the same
def generate_stress_dataset(
    base_output_dir: str = "./realistic_shm_data",
    num_files_per_folder: int = 50,
    plot_sample_for_each_type: bool = True
):
    """
    Generate realistic structural health monitoring dataset with proper variability
    """
    
    print(f"Generating realistic SHM dataset in: {os.path.abspath(base_output_dir)}")
    os.makedirs(base_output_dir, exist_ok=True)
    
    generated_file_paths = []
    
    for stress_level in STRESS_PARAMETERS.keys():
        folder_path = os.path.join(base_output_dir, stress_level.replace(" ", "_"))
        os.makedirs(folder_path, exist_ok=True)
        print(f"\nGenerating realistic '{stress_level}' signals in '{folder_path}'")
        
        sample_df_to_plot = None
        
        for i in tqdm(range(num_files_per_folder), desc=f"Generating {stress_level} files"):
            filename = f"{stress_level.replace(' ', '_')}_{i+1:04d}.txt"
            filepath = os.path.join(folder_path, filename)
            
            df = generate_realistic_signal(stress_level, filepath)
            generated_file_paths.append(filepath)
            
            if i == 0 and plot_sample_for_each_type:
                sample_df_to_plot = df
        
        if plot_sample_for_each_type and sample_df_to_plot is not None:
            plt.figure(figsize=(15, 8))
            
            # Magnitude plot
            plt.subplot(2, 1, 1)
            plt.plot(sample_df_to_plot["Frequency"], sample_df_to_plot["Magnitude"], 
                     'b-', linewidth=0.8, alpha=0.8)
            plt.title(f"Realistic SHM Signal: '{stress_level}' - Magnitude Response (Random Transient Spikes)", 
                      fontsize=12, fontweight='bold')
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Magnitude (dB)")
            plt.grid(True, alpha=0.3)
            plt.xlim(0, MAX_FREQ_HZ)
            
            # Dynamic Y-axis limit for better visualization
            y_min = sample_df_to_plot["Magnitude"].min() - 5
            y_max = sample_df_to_plot["Magnitude"].max() + 10
            plt.ylim(y_min, y_max)
            
            # Phase plot
            plt.subplot(2, 1, 2)
            plt.plot(sample_df_to_plot["Frequency"], sample_df_to_plot["Phase"], 
                     'r-', linewidth=0.8, alpha=0.8)
            plt.title(f"Phase Response", fontsize=12)
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Phase (degrees)")
            plt.grid(True, alpha=0.3)
            plt.xlim(0, MAX_FREQ_HZ)
            plt.ylim(-180, 180)
            
            plt.tight_layout()
            plt.show()
    
    print(f"\nRealistic SHM dataset generation complete!")
    print(f"Total files generated: {len(generated_file_paths)}")
    print(f"This data now includes:")
    print("- Proper modal frequencies and damping with increased per-sample randomness")
    print("- Environmental effects (temperature, humidity, wind)")
    print("- Measurement noise and artifacts")
    print("- Operational variability")
    print("- Realistic frequency content and phase relationships")
    print("- **Random, large transient spikes with random frequencies and strengths!**")
    print("- **Significant overlap between classes (making it challenging!)**")
    print("- **Global excitation factor for realistic amplitude variations**")
    
    return generated_file_paths

# ============ MAIN EXECUTION ============
if __name__ == "__main__":
    output_directory = "./data_challenging" # Changed output dir name
    num_samples = 100
    show_example_plots = True
    
    generated_files = generate_stress_dataset(
        base_output_dir=output_directory,
        num_files_per_folder=num_samples,
        plot_sample_for_each_type=show_example_plots
    )
    
    print(f"\nChallenging and realistic SHM dataset ready in: {os.path.abspath(output_directory)}")
    print("This dataset will be much more difficult to classify, requiring models to learn subtle statistical trends.")