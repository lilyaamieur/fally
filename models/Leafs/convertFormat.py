import torch
import pickle
import os

def convert_pth_to_pkl(pth_filename: str, pkl_filename: str = None):
    """
    Loads a PyTorch .pth file (assumed to be a state_dict or a complete model)
    and saves its content to a Python .pkl (pickle) file.

    Parameters:
    - pth_filename (str): The name of the input .pth file in the current directory.
    - pkl_filename (str, optional): The name of the output .pkl file.
                                    If None, it defaults to the .pth filename
                                    with the extension changed to .pkl.
    """
    
    if not os.path.exists(pth_filename):
        print(f"Error: The .pth file '{pth_filename}' does not exist in the current directory.")
        return

    if pkl_filename is None:
        # Generate default .pkl filename by changing the extension
        name_without_ext = os.path.splitext(pth_filename)[0]
        pkl_filename = f"{name_without_ext}.pkl"

    try:
        print(f"Loading data from '{pth_filename}'...")
        # Load the PyTorch .pth file
        # map_location='cpu' ensures it loads even if GPU is not available
        # It's generally good practice for loading saved models.
        loaded_data = torch.load(pth_filename, map_location='cpu')

        print(f"Saving data to '{pkl_filename}'...")
        # Save the loaded data to a .pkl file
        with open(pkl_filename, 'wb') as f:
            pickle.dump(loaded_data, f)

        print(f"Successfully converted '{pth_filename}' to '{pkl_filename}'.")

    except Exception as e:
        print(f"An error occurred during conversion: {e}")

# ============ Example Usage ============
if __name__ == "__main__":
    input_pth_file = "model.pth" # Or "signal_classifier.pth"

    output_pkl_file = None 

    convert_pth_to_pkl(input_pth_file, output_pkl_file)
