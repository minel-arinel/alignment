import logging
import numpy as np
import os
from pathlib import Path


def decode_mmap_filename_dict(basename:str) -> dict:
    """
    Extracts parameters encoded in the filename of a memory-mapped file.
    Parameters:
        basename (str): The base name of the file.
    Returns:
        dict: A dictionary containing extracted parameters such as 'd1', 'd2', 'd3', 'order', 'frames', and 'T'.

    Notes:
        - Assumes filenames are constructed with parameters separated by underscores.
        - Parameters 'd1', 'd2', 'd3', and 'frames' are expected to be numeric.
        - Parameter 'order' is expected to be a string.
        - If 'T' is not explicitly provided, it defaults to the value of 'frames'.
        - Prints a message if 'T' and 'frames' values differ.
    """
    ret = {}
    _, fn = os.path.split(basename)
    fn_base, _ = os.path.splitext(fn)
    fpart = fn_base.split('_')[1:] # First part will (probably) reference the datasets
    
    for field in ['d1', 'd2', 'd3', 'order', 'frames']:
        # look for the last index of fpart and look at the next index for the value, saving into ret
        for i in range(len(fpart) - 1, -1, -1): # Step backwards through the list; defensive programming
            if field == fpart[i]:
                if field == 'order': # a string
                    ret[field] = fpart[i + 1] # Assume no filenames will be constructed to end with a key and not a value
                else: # numeric
                    ret[field] = int(fpart[i + 1]) # Assume no filenames will be constructed to end with a key and not a value
    
    if fpart[-1] != '':
        ret['T'] = int(fpart[-1])
    
    if 'T' in ret and 'frames' in ret and ret['T'] != ret['frames']:
        print(f"D: The value of 'T' {ret['T']} differs from 'frames' {ret['frames']}")
    
    if 'T' not in ret and 'frames' in ret:
        ret['T'] = ret['frames']
    
    return ret


def caiman_datadir() -> str:
    """
    Returns the directory for CaImAn data, which can be user-configurable.
    Prioritizes the 'CAIMAN_DATA' environment variable if set.
    """
    if "CAIMAN_DATA" in os.environ:
        return os.environ["CAIMAN_DATA"]
    else:
        return os.path.join(os.path.expanduser("~"), "caiman_data")


def get_tempdir() -> str:
    """
    Returns the directory where CaImAn can store temporary files, such as memmap files.
    The directory is determined by the following order of precedence:
    1. If the environment variable 'CAIMAN_TEMP' is set and points to an existing directory, that directory is used.
    2. If 'CAIMAN_TEMP' is set but points to a nonexistent directory, a warning is logged, and the default directory is used.
    3. If 'CAIMAN_TEMP' is not set, a default 'temp' directory under the CaImAn data directory is used. If this directory does not exist, it is created.
    Returns:
        str: Path to the directory for storing temporary files.
    """
    logger = logging.getLogger("caiman")

    if 'CAIMAN_TEMP' in os.environ:
        if os.path.isdir(os.environ['CAIMAN_TEMP']):
            return os.environ['CAIMAN_TEMP']
        else:
            logger.warning(f"CAIMAN_TEMP is set to nonexistent directory {os.environ['CAIMAN_TEMP']}. Ignoring")
    temp_under_data = os.path.join(caiman_datadir(), "temp")
    if not os.path.isdir(temp_under_data):
        logger.warning(f"Default temporary dir {temp_under_data} does not exist, creating")
        os.makedirs(temp_under_data)
    return temp_under_data


def fn_relocated(fn:str, force_temp:bool=False) -> str:
    """
    Relocates a filename to a temporary directory if no path is provided.

    Args:
        fn (str): The filename to be relocated.
        force_temp (bool): If True, forces relocation to the temporary directory even if a path is provided.

    Returns:
        str: The absolute pathname in the temporary directory or the original filename/path.
    """
    if os.path.split(fn)[0] == '': # No path stuff
        return os.path.join(get_tempdir(), fn)
    elif force_temp:
        return os.path.join(get_tempdir(), os.path.split(fn)[1])
    else:
        return fn


def prepare_shape(mytuple:tuple) -> tuple:
    """
    Convert elements of the shape tuple to np.uint64 to prevent overflow in numpy operations.
    """
    if not isinstance(mytuple, tuple):
        raise Exception("Internal error: prepare_shape() passed a non-tuple")
    return tuple(map(lambda x: np.uint64(x), mytuple))


def load_memmap(filename, mode):
    """
    Load a memory-mapped file.
    Parameters:
        filename (str): Path of the file to be loaded.
        mode (str): File access mode ('r', 'r+', 'w+').
    Returns:
        np.memmap: Memory-mapped variable.
        tuple: Frame dimensions.
        int: Number of frames.
    Raises:
        ValueError: If the file extension is not '.mmap'.
    """
    logger = logging.getLogger("caiman")
    if Path(filename).suffix != '.mmap':
        logger.error(f"Unknown extension for file {filename}")
        raise ValueError(f'Unknown file extension for file {filename} (should be .mmap)')
    
    decoded_fn = decode_mmap_filename_dict(filename)
    d1		= decoded_fn['d1']
    d2		= decoded_fn['d2']
    d3		= decoded_fn['d3']
    T		= decoded_fn['T']
    order  	= decoded_fn['order']

    #d1, d2, d3, T, order = int(fpart[-9]), int(fpart[-7]), int(fpart[-5]), int(fpart[-1]), fpart[-3]

    filename = fn_relocated(filename)
    Yr = np.memmap(filename, mode=mode, shape=prepare_shape((d1 * d2 * d3, T)), 
                   dtype=np.float32, order=order)
    if d3 == 1:
        return (Yr, (d1, d2), T)
    else:
        return (Yr, (d1, d2, d3), T)
    

def caiman_memmap_reader(path: str, **kwargs) -> np.memmap:
    """
    Reads a memory-mapped file and reshapes it for further processing.
    Parameters:
        path (str): The file path to the memory-mapped file.
        **kwargs: Additional arguments passed to the load_memmap function.
    Returns:
        np.memmap: The reshaped memory-mapped array.
    """
    Yr, dims, T = load_memmap(path, **kwargs)
    return np.reshape(Yr.T, [T] + list(dims), order="F")


def minmax_scaler(arr, vmin=0, vmax=1):
    """
    Scales the input array to a specified range [vmin, vmax].
    Parameters:
        arr (numpy.ndarray): Input array to be scaled.
        vmin (float, optional): Minimum value of the scaled range. Default is 0.
        vmax (float, optional): Maximum value of the scaled range. Default is 1.
    Returns:
        numpy.ndarray: Scaled array with values in the range [vmin, vmax].
    """
    arr_min, arr_max = arr.min(), arr.max()
    return ((arr - arr_min) / (arr_max - arr_min)) * (vmax - vmin) + vmin