import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm  # For progress bars

# ========== CONFIG ==========
BASE_DIR = "./data"
# UPDATED FOLDERS for the new dataset categories
FOLDERS = ["healthy", "heavy_stress", "light_stress", "medium_stress", ""]
N_SAMPLES = 3  # Number of samples to display per category
MAX_FILES_TO_PROCESS = 100  # Safety limit for processing files per folder
COLORS = ['#4CAF50', '#FFC107', '#FF9800', '#F44336']  # Material design colors
sns.set(style="whitegrid", font_scale=0.8)  # Set seaborn style with proper scaling

# ========== UTILS ==========
def read_signal_file(filepath):
    """Read signal file with robust header handling"""
    try:
        # First try reading assuming the first row might be header, but read as data
        # Then convert to numeric and drop non-numeric rows
        df = pd.read_csv(
            filepath,
            sep="\t",
            encoding="latin1",
            header=None, # Treat everything as data initially
            names=["Frequency", "Magnitude", "Phase"],
            on_bad_lines="skip"
        )

        # Convert to numeric, coercing errors to NaN
        df = df.apply(pd.to_numeric, errors='coerce').dropna()

        if df.empty:
            raise ValueError("No valid numeric data found after parsing")
            
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
                 and os.path.getsize(os.path.join(folder, f)) > 10]  # Minimum 10 bytes to avoid empty/corrupt files
        return files[:MAX_FILES_TO_PROCESS] # Limit the number of files processed per folder
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
    # Adjust figsize based on the number of samples and folders if needed
    fig, axs = plt.subplots(
        len(FOLDERS), N_SAMPLES,
        figsize=(18, 4 * len(FOLDERS)), # Dynamic height based on number of folders
        squeeze=False,
        constrained_layout=True
    )
    fig.suptitle("Signal Visualization per Condition", fontsize=14, y=1.02)

    for row, folder in enumerate(tqdm(FOLDERS, desc="Processing folders for individual plots")):
        folder_path = os.path.join(BASE_DIR, folder)

        # Check if folder exists
        if not os.path.exists(folder_path):
            print(f"\nFolder '{folder_path}' not found, skipping...")
            for col in range(N_SAMPLES):
                axs[row, col].set_title(f"{folder}\n(Folder not found)", fontsize=10, color='red')
                axs[row, col].axis('off')
            continue

        filenames = get_txt_files(folder_path)

        if not filenames:
            print(f"\nNo valid .txt files found in {folder_path}, skipping...")
            for col in range(N_SAMPLES):
                axs[row, col].set_title(f"{folder}\n(No files found)", fontsize=10, color='red')
                axs[row, col].axis('off')
            continue
            
        # Select N_SAMPLES files, or fewer if not enough exist
        files_to_plot = filenames[:N_SAMPLES]

        for col, fname in enumerate(files_to_plot):
            path = os.path.join(folder_path, fname)
            df = read_signal_file(path)

            if df is not None and not df.empty:
                plot_signal(
                    axs[row, col],
                    df["Frequency"],
                    df["Magnitude"],
                    title=f"{folder}\n{os.path.basename(fname)}", # Use basename for cleaner title
                    color=COLORS[row % len(COLORS)] # Use modulo for color safety
                )
            else:
                axs[row, col].set_title(f"{folder}\n{os.path.basename(fname)}\n(Data error)", fontsize=10, color='orange')
                axs[row, col].axis('off')  # Hide empty/error subplots
        
        # Hide any remaining empty subplots if fewer than N_SAMPLES files were found
        for col_fill in range(len(files_to_plot), N_SAMPLES):
            axs[row, col_fill].axis('off')

    plt.show()

# ========== OVERLAPPING PLOT (ONE SAMPLE PER CLASS) ==========
def plot_sample_comparison():
    """Plot one representative sample from each class"""
    plt.figure(figsize=(12, 7))

    for i, folder in enumerate(FOLDERS):
        folder_path = os.path.join(BASE_DIR, folder)

        if not os.path.exists(folder_path):
            print(f"\nFolder '{folder_path}' not found, skipping sample comparison for {folder}...")
            continue

        files = get_txt_files(folder_path)

        if not files:
            print(f"\nNo valid .txt files found in {folder_path}, skipping sample comparison for {folder}...")
            continue

        file_path = os.path.join(folder_path, files[0]) # Take the first valid file
        df = read_signal_file(file_path)

        if df is not None and not df.empty:
            plt.plot(
                df["Frequency"],
                df["Magnitude"],
                label=folder,
                color=COLORS[i % len(COLORS)], # Use modulo for color safety
                alpha=0.8,
                linewidth=2
            )
        else:
            print(f"Could not read valid data from first file in {folder}: {file_path}")

    plt.title("Frequency-Magnitude Comparison (1 Sample per Class)", pad=20)
    plt.xlabel("Frequency (Hz)", fontsize=10)
    plt.ylabel("Magnitude (dB)", fontsize=10)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# ========== AVERAGED PLOT (ALL FILES PER CLASS) ==========
def plot_average_curves():
    """Plot averaged signals for each class with standard deviation fill"""
    plt.figure(figsize=(12, 7))

    for i, folder in enumerate(tqdm(FOLDERS, desc="Averaging signals")):
        folder_path = os.path.join(BASE_DIR, folder)

        if not os.path.exists(folder_path):
            print(f"\nFolder '{folder_path}' not found, skipping averaging for {folder}...")
            continue

        files = get_txt_files(folder_path)
        valid_dfs = []

        for file in tqdm(files, desc=f"Reading {folder}", leave=False):
            df = read_signal_file(os.path.join(folder_path, file))
            if df is not None and not df.empty:
                valid_dfs.append(df)

        if not valid_dfs:
            print(f"\nNo valid data in {folder_path} for averaging, skipping...")
            continue

        # Align all signals to the same frequency range based on the minimum length
        # This handles cases where files might have different numbers of rows.
        # It's crucial for correct averaging.
        min_len = min(len(df) for df in valid_dfs)
        aligned_mags = np.array([df["Magnitude"].values[:min_len] for df in valid_dfs])
        freqs = valid_dfs[0]["Frequency"].values[:min_len] # Assume frequencies are consistent up to min_len

        avg_mag = np.mean(aligned_mags, axis=0)
        std_mag = np.std(aligned_mags, axis=0)

        plt.plot(freqs, avg_mag, label=folder, color=COLORS[i % len(COLORS)], linewidth=2)
        plt.fill_between(
            freqs,
            avg_mag - std_mag,
            avg_mag + std_mag,
            color=COLORS[i % len(COLORS)],
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
    print("Starting signal EDA for bearing, misalignment, normal, and unbalance conditions...")

    # Validate BASE_DIR exists
    if not os.path.exists(BASE_DIR):
        print(f"Error: BASE_DIR '{BASE_DIR}' does not exist. Please create it and place your data folders inside.")
    else:
        # Check if any of the expected folders exist
        found_any_folder = False
        for folder in FOLDERS:
            if os.path.exists(os.path.join(BASE_DIR, folder)):
                found_any_folder = True
                break
        
        if not found_any_folder:
            print(f"Warning: None of the expected data folders ({', '.join(FOLDERS)}) found in '{BASE_DIR}'. No plots will be generated.")
            print("Please ensure your data is organized like: BASE_DIR/bearing/, BASE_DIR/misalignment/, etc.")

    try:
        plot_individual_samples()
        plot_sample_comparison()
        plot_average_curves()
    except Exception as e:
        print(f"\nAn error occurred during EDA: {str(e)}")
    finally:
        print("EDA complete.")

