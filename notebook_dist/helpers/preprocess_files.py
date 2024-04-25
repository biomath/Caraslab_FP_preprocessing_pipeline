from re import split, search
from glob import glob
from datetime import datetime
from platform import system
from os.path import sep
# Tweak the regex file separator for cross-platform compatibility
if system() == 'Windows':
    REGEX_SEP = sep * 2
else:
    REGEX_SEP = sep
import json

from pandas import DataFrame, read_csv


def preprocess_files(memory_path, settings_dict):
    # Match spike_times with appropriate key_files
    keys_path = settings_dict['KEYS_PATH']

    # Split path name to get subject, session and unit ID for prettier output
    split_memory_path = split(REGEX_SEP, memory_path)  # split path
    recording_id = split_memory_path[-1][:-4]  # Example id: SUBJ-ID-104_FP-Aversive-AM-210707-110339_dff

    split_timestamps_name = split("_*_", recording_id)[1]  # split timestamps
    if '1IFC' in split_timestamps_name or '1IFC' in settings_dict['EXPERIMENT_TYPE']:
        cur_date = split("-*-", split_timestamps_name)[2]
        cur_timestamp = split("-*-", split_timestamps_name)[3]
    elif 'Aversive' in split_timestamps_name or 'Passive' in split_timestamps_name:
        cur_date = split("-*-", split_timestamps_name)[3]
        cur_timestamp = split("-*-", split_timestamps_name)[4]
    else:
        print('Experiment type  not found in file name')
        return

    subject_id = split("_*_", recording_id)[0]
    subj_date = subject_id + '_' + cur_date

    # These are in alphabetical order. Must sort by date_trial or match with file
    key_path_info = glob(keys_path + sep + subject_id + '*' +
                         cur_date + '*' + cur_timestamp + "*_trialInfo.csv")

    if len(key_path_info) == 0:
        print("Key not found for " + recording_id)
        return

    # Load key and spout files
    info_key_times = read_csv(key_path_info[0])

    if 'Aversive' in key_path_info[0]:
        key_path_spout = glob(keys_path + sep + subject_id + '*' +
                              cur_date + '*' + cur_timestamp + "*spoutTimestamps.csv")
        spout_key_times = read_csv(key_path_spout[0])
        trial_types = ['Hit (shock)', 'Hit (no shock)', 'Miss (shock)', 'Miss (no shock)', 'False alarm']
    elif 'Passive' in key_path_info[0]:
        spout_key_times = None
        trial_types = ['Passive', ]
    elif '1IFC' in key_path_info[0]:
        spout_key_times = None
        trial_types = ['Hit', 'Miss', 'Reject', 'False alarm']
    else:
        print('Experiment type not found in key file name')
        return

    return subj_date, info_key_times, spout_key_times, trial_types