import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm  # For progress bars

# ========== CONFIG ==========
BASE_DIR = "./"
FOLDERS = ["Healthy", "Light Damage", "Medium Damage", "Severe Damage"]
N_SAMPLES = 3  # Number of samples to display per category
MAX_FILES_TO_PROCESS = 100  # Safety limit for processing files
COLORS = ['#4CAF50', '#FFC107', '#FF9800', '#F44336']  # Material design colors
sns.set(style="whitegrid", font_scale=0.8)  # Set seaborn style with proper scaling

# ========== UTILS ==========
def read_signal_file(filepath):
    """Read signal file with robust header handling"""
    try:
        # First try reading without header
        df = pd.read_csv(
            filepath, 
            sep="\t", 
            encoding="latin1", 
            header=None,
            names=["Frequency", "Magnitude", "Phase"],
            skiprows=1,  # Skip potential header row
            on_bad_lines="skip"
        )
        
        # If first value is string (header got through), skip that row
        if pd.api.types.is_string_dtype(df.iloc[0,0]):
            df = pd.read_csv(
                filepath,
                sep="\t",
                encoding="latin1",
                header=None,
                names=["Frequency", "Magnitude", "Phase"],
                skiprows=1,
                on_bad_lines="skip"
            )
            
        # Convert to numeric, coercing errors to NaN
        df = df.apply(pd.to_numeric, errors='coerce').dropna()
        
        if df.empty:
            raise ValueError("No valid numeric data found")
            
        return df
        
    except Exception as e:
        print(f"\nError processing {os.path.basename(filepath)}: {str(e)}")
        return None

def get_txt_files(folder):
    """Get text files with size validation"""
    try:
        files = [f for f in sorted(os.listdir(folder)) 
                if f.endswith(".txt") 
                and not f.startswith('.')
                and os.path.getsize(os.path.join(folder, f)) > 10]  # Minimum 10 bytes
        return files[:MAX_FILES_TO_PROCESS]
    except Exception as e:
        print(f"\nError accessing {folder}: {str(e)}")
        return []

def plot_signal(ax, x, y, title="", color='b', alpha=0.7):
    """Standardized signal plotting"""
    ax.plot(x, y, color=color, alpha=alpha, linewidth=1.5)
    ax.set_title(title, fontsize=10, pad=5)
    ax.set_xlabel("Frequency (Hz)", fontsize=8)
    ax.set_ylabel("Magnitude (dB)", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=8)

# ========== VISUALIZE INDIVIDUAL SAMPLES ==========
def plot_individual_samples():
    """Plot individual samples in a grid layout"""
    fig, axs = plt.subplots(
        len(FOLDERS), N_SAMPLES, 
        figsize=(18, 12), 
        squeeze=False,
        constrained_layout=True
    )
    fig.suptitle("Signal Visualization per Damage Level", fontsize=14, y=1.02)
    
    for row, folder in enumerate(tqdm(FOLDERS, desc="Processing folders")):
        folder_path = os.path.join(BASE_DIR, folder)
        filenames = get_txt_files(folder_path)[:N_SAMPLES]
        
        if not filenames:
            print(f"\nNo files found in {folder}, skipping...")
            continue
            
        for col, fname in enumerate(filenames):
            path = os.path.join(folder_path, fname)
            df = read_signal_file(path)
            
            if df is not None:
                plot_signal(
                    axs[row, col],
                    df["Frequency"],
                    df["Magnitude"],
                    title=f"{folder}\n{fname}",
                    color=COLORS[row]
                )
            else:
                axs[row, col].axis('off')  # Hide empty subplots
    
    plt.show()

# ========== OVERLAPPING PLOT (ONE SAMPLE PER CLASS) ==========
def plot_sample_comparison():
    """Plot one representative sample from each class"""
    plt.figure(figsize=(12, 7))
    
    for i, folder in enumerate(FOLDERS):
        folder_path = os.path.join(BASE_DIR, folder)
        files = get_txt_files(folder_path)
        
        if not files:
            print(f"\nNo files found in {folder}, skipping...")
            continue
            
        file_path = os.path.join(folder_path, files[0])
        df = read_signal_file(file_path)
        
        if df is not None:
            plt.plot(
                df["Frequency"], 
                df["Magnitude"], 
                label=folder, 
                color=COLORS[i],
                alpha=0.8,
                linewidth=2
            )
    
    plt.title("Frequency-Magnitude Comparison (1 Sample per Class)", pad=20)
    plt.xlabel("Frequency (Hz)", fontsize=10)
    plt.ylabel("Magnitude (dB)", fontsize=10)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# ========== AVERAGED PLOT (ALL FILES PER CLASS) ==========
def plot_average_curves():
    """Plot averaged signals for each class"""
    plt.figure(figsize=(12, 7))
    
    for i, folder in enumerate(tqdm(FOLDERS, desc="Averaging signals")):
        folder_path = os.path.join(BASE_DIR, folder)
        files = get_txt_files(folder_path)
        valid_dfs = []
        
        for file in tqdm(files, desc=folder, leave=False):
            df = read_signal_file(os.path.join(folder_path, file))
            if df is not None:
                valid_dfs.append(df)
        
        if not valid_dfs:
            print(f"\nNo valid data in {folder}, skipping...")
            continue
            
        # Align all signals to the same frequency range
        min_len = min(len(df) for df in valid_dfs)
        aligned_mags = np.array([df["Magnitude"].values[:min_len] for df in valid_dfs])
        freqs = valid_dfs[0]["Frequency"].values[:min_len]
        
        avg_mag = np.mean(aligned_mags, axis=0)
        std_mag = np.std(aligned_mags, axis=0)
        
        plt.plot(freqs, avg_mag, label=folder, color=COLORS[i], linewidth=2)
        plt.fill_between(
            freqs, 
            avg_mag - std_mag, 
            avg_mag + std_mag, 
            color=COLORS[i], 
            alpha=0.2
        )
    
    plt.title("Average Frequency-Magnitude Curves with Standard Deviation", pad=20)
    plt.xlabel("Frequency (Hz)", fontsize=10)
    plt.ylabel("Magnitude (dB)", fontsize=10)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# ========== MAIN EXECUTION ==========
if __name__ == "__main__":
    print("Starting signal visualization...")
    
    try:
        plot_individual_samples()
        plot_sample_comparison()
        plot_average_curves()
    except Exception as e:
        print(f"\nError in main execution: {str(e)}")
    finally:
        print("Visualization complete.")
        