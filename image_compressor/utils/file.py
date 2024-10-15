import numpy as np

def format_file_size(size_bytes):
    """Convert bytes to a more human-readable format."""
    if size_bytes == 0:
        return "0 Bytes"
    size_name = ("Bytes", "KB", "MB", "GB", "TB")
    i = int(np.floor(np.log(size_bytes) / np.log(1024)))  # Calculate the index for the size name
    p = np.power(1024, i)  # Calculate the power of 1024
    s = round(size_bytes / p, 2)  # Calculate the size in the appropriate unit
    return f"{s} {size_name[i]}"
