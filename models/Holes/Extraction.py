import numpy as np
import matplotlib.pyplot as plt

# ========== CONFIG ==========
def generate_synthetic_data(num_rows=6403):
    frequencies = np.linspace(0, 1000, num_rows)

    magnitudes = np.zeros(num_rows)
    magnitudes += 5

    # Refined parameters for synthetic peaks to better match the graph
    # Peak 1 (more prominent and wider, starting around 50-100Hz)
    amp1, freq1, width1 = 45, 100, 30
    magnitudes += amp1 * np.exp(-((frequencies - freq1)**2) / (2 * width1**2))

    # Peak 2 (around 200-250Hz)
    amp2, freq2, width2 = 30, 250, 15
    magnitudes += amp2 * np.exp(-((frequencies - freq2)**2) / (2 * width2**2))

    # Peak 3 (around 300-350Hz)
    amp3, freq3, width3 = 55, 350, 20
    magnitudes += amp3 * np.exp(-((frequencies - freq3)**2) / (2 * width3**2))

    # Peak 4 (around 500Hz)
    amp4, freq4, width4 = 40, 500, 25
    magnitudes += amp4 * np.exp(-((frequencies - freq4)**2) / (2 * width4**2))

    # Peak 5 (around 700Hz)
    amp5, freq5, width5 = 30, 700, 20
    magnitudes += amp5 * np.exp(-((frequencies - freq5)**2) / (2 * width5**2))

    # Peak 6 (around 900Hz, broader)
    amp6, freq6, width6 = 28, 900, 40
    magnitudes += amp6 * np.exp(-((frequencies - freq6)**2) / (2 * width6**2))

    magnitudes += 8 * np.sin(frequencies / 150)
    magnitudes += np.random.normal(0, 1.2, num_rows)

    magnitudes[magnitudes < 0] = 0.1

    # Adjusted phase shifts for better resemblance
    phases = np.interp(frequencies,
                       [0, freq1, freq1 + width1, freq2, freq2 + width2,
                        freq3, freq3 + width3, freq4, freq4 + width4,
                        freq5, freq5 + width5, freq6, freq6 + width6, 1000],
                       [0, -60, -120, -150, -220, -260, -330, -380, -450,
                        -490, -560, -600, -670, -700])

    phases += np.random.normal(0, 4, num_rows)

    return frequencies, magnitudes, phases

def save_to_file(filename, frequencies, magnitudes, phases):
    with open(filename, 'w') as f:
        f.write("Frequency\tMagnitude\tPhase\n")
        for i in range(len(frequencies)):
            f.write(f"{frequencies[i]:.4f}\t{magnitudes[i]:.4f}\t{phases[i]:.4f}\n")
    print(f"Data saved to {filename}")
    print(f"{len(frequencies)} Rows")


# ========== Visuals ==========
def plot_data(frequencies, magnitudes, phases):
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(frequencies, magnitudes, color='lightgreen')
    plt.title('Synthetic Frequency Response (Magnitude)')
    plt.xlabel('Frequency')
    plt.ylabel('Magnitude')
    plt.grid(True, linestyle='-', alpha=0.6)
    plt.tight_layout()

    plt.subplot(2, 1, 2)
    plt.plot(frequencies, phases, color='skyblue')
    plt.title('Synthetic Frequency Response (Phase)')
    plt.xlabel('Frequency')
    plt.ylabel('Phase (degrees)')
    plt.grid(True, linestyle='-', alpha=0.6)
    plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    freq, mag, pha = generate_synthetic_data()

    save_to_file("Observation.txt", freq, mag, pha)

    plot_data(freq, mag, pha)
