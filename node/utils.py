import os
import numpy as np
def save_labels(labels, path):
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Save the labels to the specified path
    np.savetxt(path, labels, fmt='%d')