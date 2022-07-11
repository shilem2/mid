import pickle
import blosc
import numpy as np


def save_compressed_pickle(data, filename):

    pickled = pickle.dumps(data)  # binary test_data
    compressed = blosc.compress(pickled)
    with open(filename, 'wb') as f:
        f.write(compressed)

    return

def load_compressed_pickle(filename):

    with open(filename, 'rb') as f:
        compressed = f.read()
    pickled = blosc.decompress(compressed)
    data = pickle.loads(pickled)

    return data

def convert_array_to_json(x):
    if hasattr(x, "tolist"):  # numpy arrays have this
        return {"$array": x.tolist()}  # Make a tagged object
    raise TypeError(x)


def convert_json_to_array(x):
    if len(x) == 1:  # Might be a tagged object...
        key, value = next(iter(x.items()))  # Grab the tag and value
        if key == "$array":  # If the tag is correct,
            return np.array(value)  # cast back to array
    return x
