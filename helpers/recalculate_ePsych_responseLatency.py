"""Recomputes AM-detection task response latencies from spout-offset timestamps and rewrites trialInfo.csv in place."""
import numpy as np
from re import split
from os.path import sep
from copy import deepcopy
import platform
from helpers.preprocess_files import load_session_files

# Tweak the regex file separator for cross-platform compatibility
if platform.system() == 'Windows':
    REGEX_SEP = sep * 2
else:
    REGEX_SEP = sep


def recalculate_ePsych_responseLatency(input_list):
    """Recalculate response latency from spout-offset timestamps for the AM-detection (aversive) task.

    Older RPvds circuits did not correctly compute response latency online,
    so this derives it after the fact from spout on/off timestamps and
    overwrites ``RespLatency`` in each session's ``trialInfo.csv``. Passive
    sessions (identified by ``'passive'`` in the file path) are skipped.
    Latency for Hit/false-alarm trials is the time from trial onset to the
    last spout-offset event during the trial window; for Miss trials it
    falls back to spout-offset events during the shock window
    (``SETTINGS_DICT['SHOCK_START_END']``) if none occurred during the trial
    itself. Trials with no qualifying spout-offset event get ``NaN``.

    Parameters
    ----------
    input_list : tuple of (list of str, dict)
        A single positional argument packing ``(session_date_paths,
        SETTINGS_DICT)`` — this tuple-argument convention (rather than
        separate keyword arguments) exists so the function can be used
        directly as the worker in ``multiprocessing.Pool.map``, which only
        passes one positional argument per call.

        session_date_paths : list of str
            Recording key-file paths (as accepted by
            :func:`helpers.preprocess_files.load_session_files`) to process.
        SETTINGS_DICT : dict
            Pipeline settings; must include ``'TRIAL_OR_RESPONSE_ALIGNED'``
            (str or list — if a list, only its first element is used),
            ``'SHOCK_START_END'``, and ``'KEYS_PATH'`` (used as both the
            source and, by default, the output directory for the rewritten
            ``trialInfo.csv`` files).

    Returns
    -------
    None
        This function has a side effect only: it overwrites each session's
        ``trialInfo.csv`` in place with a recalculated ``RespLatency`` column.
    """

    (session_date_paths, SETTINGS_DICT) = input_list

    run_settings = deepcopy(SETTINGS_DICT)

    # Add this dummy fix here just so that this code doesn't break when this variable is a list
    if type(run_settings['TRIAL_OR_RESPONSE_ALIGNED']) == str:
        pass
    else:  # must be list
        assert type(
            run_settings['TRIAL_OR_RESPONSE_ALIGNED']) == list, 'TRIAL_OR_RESPONSE_ALIGNED must be a string or a list'
        run_settings['TRIAL_OR_RESPONSE_ALIGNED'] = run_settings['TRIAL_OR_RESPONSE_ALIGNED'][0]

    # Load globals
    shock_start_end = run_settings['SHOCK_START_END']

    output_path = run_settings['KEYS_PATH']
    save_dir = output_path

    # Uncomment below if you don't want to overwrite the original files
    # save_dir = output_path + sep + 'new_respLatencies'
    # makedirs(save_dir, exist_ok=True)

    for recording_path in session_date_paths:
        # Automatically skip passive files here for obvious reasons :)
        if 'passive' in recording_path.lower():
            continue

        subj_date, info_key_times, spout_key_times, _, _ = load_session_files(recording_path, run_settings)

        if info_key_times is None:
            print('Something weird with: ' + recording_path + '. Could not gather trial info \n\n')
            continue

        if spout_key_times is None:
            print('Something weird with: ' + recording_path + '. Could not gather spout offset times\n\n')
            continue

        spout_offsets = spout_key_times['Spout_offset'].values

        new_latencies = np.zeros(len(info_key_times))

        for row_idx, row_slice in info_key_times.iterrows():
            if (row_slice['Hit'] == 1) | (row_slice['FA'] == 1):
                cur_onset = row_slice['Trial_onset']
                cur_offset = row_slice['Trial_offset']
                cur_spout_offsets = spout_offsets[(spout_offsets >= cur_onset) & (spout_offsets < cur_offset)]
                if len(cur_spout_offsets) == 0:  # Sometimes this is not registered properly in RZ6
                    print('Spout offset not registered properly in: ' + recording_path +
                          '\nTrialID: ' + str(row_slice['TrialID']) + '\n\n')
                    new_latencies[row_idx] = np.nan
                else:
                    last_offset = cur_spout_offsets[-1]  # Last offset probably triggered the outcome
                    new_latencies[row_idx] = last_offset - cur_onset
            elif row_slice['Miss'] == 1:
                # If miss trial:
                # 1. Look for spout offsets during the trial. These trials can be handled separately since the animal
                #   might have detected the AM sound but failed to stay off spout for some reason
                # 2. If no spout offsets during the trial were found, look for offsets during the shock period. If
                #   none are found, return NaN
                cur_onset = row_slice['Trial_onset']
                cur_offset = row_slice['Trial_offset']

                cur_spout_offsets = spout_offsets[(spout_offsets >= cur_onset) & (spout_offsets < cur_offset)]
                if len(cur_spout_offsets) == 0:
                    cur_spout_offsets = spout_offsets[(spout_offsets >= (cur_onset + shock_start_end[0])) &
                                                      (spout_offsets < (cur_onset + shock_start_end[1]))]

                if len(cur_spout_offsets) == 0:  # Either animal did not withdraw with shock or this was a non-shocked miss
                    new_latencies[row_idx] = np.nan
                else:
                    last_offset = cur_spout_offsets[-1]  # Get the last offset
                    new_latencies[row_idx] = last_offset - cur_onset
            else:
                new_latencies[row_idx] = np.nan

        # Replace dummy latencies
        info_key_times['RespLatency'] = new_latencies

        # Save new file
        info_key_times.to_csv(save_dir + sep + split(REGEX_SEP, recording_path)[-1], index=False)
