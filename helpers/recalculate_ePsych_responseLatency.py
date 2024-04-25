import numpy as np
from re import split
from os.path import sep
from os import makedirs
import platform
from helpers.preprocess_files import preprocess_files

# Tweak the regex file separator for cross-platform compatibility
if platform.system() == 'Windows':
    REGEX_SEP = sep * 2
else:
    REGEX_SEP = sep


def recalculate_ePsych_responseLatency(input_list):
    """
    Recalculates response latency based on spout offset responses for the AM detection task
    Older RPvds circuits did not properly calculate response latencies

    :param input_list: list of inputs used by this function. Convoluted because
                       I just copied the structure from the multiprocessing functions
    :return: None; alters the trialInfo.csv files
    """

    (session_date_paths, SETTINGS_DICT) = input_list

    # Load globals
    shock_start_end = SETTINGS_DICT['SHOCK_START_END']

    output_path = SETTINGS_DICT['KEYS_PATH']
    save_dir = output_path + sep + 'new_respLatencies'
    makedirs(save_dir, exist_ok=True)

    for recording_path in session_date_paths:
        # Automatically skip passive files here for obvious reasons :)
        if 'Passive' in recording_path:
            continue

        subj_date, info_key_times, spout_key_times, trial_types = preprocess_files(recording_path, SETTINGS_DICT)

        spout_offsets = spout_key_times['Spout_offset'].values

        new_latencies = np.zeros(len(info_key_times))
        for row_idx, row_slice in info_key_times.iterrows():
            if (row_slice['Hit'] == 1) | (row_slice['FA'] == 1):
                cur_onset = row_slice['Trial_onset']
                cur_offset = row_slice['Trial_offset']
                cur_spout_offsets = spout_offsets[(spout_offsets >= cur_onset) & (spout_offsets < cur_offset)]
                if len(cur_spout_offsets) == 0:  # Sometimes this is not registered properly in RZ6
                    print('Spout offset not registered properly in: ' + recording_path +
                          '\nTrialID: ' + str(row_slice['TrialID']))
                    new_latencies[row_idx] = np.nan
                else:
                    last_offset = cur_spout_offsets[-1]  # Last offset probably triggered the outcome
                    new_latencies[row_idx] = last_offset - cur_onset
            elif (row_slice['Miss'] == 1) & (row_slice['ShockFlag'] == 1):
                # If miss+shock trial get the offset during the shock
                cur_onset = row_slice['Trial_onset']
                cur_offset = row_slice['Trial_offset']
                cur_spout_offsets = spout_offsets[(spout_offsets >= (cur_onset + shock_start_end[0])) &
                                                  (spout_offsets < (cur_offset + shock_start_end[1]))]
                if len(cur_spout_offsets) == 0:  # Sometimes this is not registered properly in RZ6
                    print('Spout offset not registered properly in: ' + recording_path +
                          '\nTrialID: ' + str(row_slice['TrialID']))
                    new_latencies[row_idx] = np.nan
                else:
                    last_offset = cur_spout_offsets[0]  # Let's get the first offset during shock period
                    new_latencies[row_idx] = last_offset - cur_onset
            else:
                new_latencies[row_idx] = np.nan

        # Replace dummy latencies
        info_key_times['RespLatency'] = new_latencies * 1000  # Convert to msec to match new RPvds output

        # Save new file
        info_key_times.to_csv(save_dir + sep + split(REGEX_SEP, recording_path)[-1])