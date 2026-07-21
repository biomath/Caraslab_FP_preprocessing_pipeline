"""Helper for serializing pipeline data structures (including numpy values) to a JSON file."""
from os import sep, makedirs
from helpers.NumpyEncoder import NumpyEncoder
import json


def write_json(cur_data, output_path, json_filename):
    """Write a data structure to disk as a pretty-printed JSON file.

    Creates ``output_path`` if it does not already exist. Numpy scalars and
    arrays anywhere in ``cur_data`` are serialized via :class:`~helpers.NumpyEncoder.NumpyEncoder`.

    Parameters
    ----------
    cur_data : dict or list
        JSON-serializable data structure to write; may contain numpy
        integers, floats, or arrays.
    output_path : str
        Directory to write the file into. Created (including parents) if missing.
    json_filename : str
        Filename (including ``.json`` extension) to write within ``output_path``.

    Returns
    -------
    None
    """
    makedirs(output_path, exist_ok=True)
    # Save as json file
    with open(output_path + sep + json_filename, 'w') as cur_json:
        cur_json.write(json.dumps(cur_data, cls=NumpyEncoder, indent=4))