"""Loads a batch of per-session JSON files, with optional filename-based include/exclude filters."""
import json
import pandas as pd


def get_JSON_data(json_filenames, sessions_to_run=None, sessions_to_exclude=None):
    """Load a list of ``*_sessionData.json`` files, optionally filtering by filename.

    Parameters
    ----------
    json_filenames : list of str
        Paths to the JSON files to load.
    sessions_to_run : list of str or str, optional
        Substrings to match against each entry of ``json_filenames``; only
        files containing at least one of these substrings are loaded. If a
        ``str`` is passed instead of a list, it is treated as a path to a CSV
        file with a ``'Unit'`` column, whose values are used as the substring
        set. If ``None`` (default), no inclusion filtering is applied.
    sessions_to_exclude : list of str, optional
        Substrings to match against each entry of ``json_filenames``; files
        containing any of these substrings are skipped. If ``None``
        (default), no exclusion filtering is applied.

    Returns
    -------
    list of dict
        One parsed JSON dict per file that passed the include/exclude
        filters, in the same order as the (filtered) ``json_filenames``.
    """
    dict_list = []
    if isinstance(sessions_to_run, str):  # file path
        sessions_file = pd.read_csv(sessions_to_run)
        sessions_to_run = set(sessions_file['Unit'].values)

    for file_name in json_filenames:
        if sessions_to_run is not None:
            if not any(chosen for chosen in sessions_to_run if chosen in file_name):
                continue

        if sessions_to_exclude is not None:
            if any(chosen for chosen in sessions_to_exclude if chosen in file_name):
                continue

        # Open JSON
        with open(file_name, 'r') as json_file:
            cur_dict = json.load(json_file)
            dict_list.append(cur_dict)

    return dict_list
