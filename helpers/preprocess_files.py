from re import split
from glob import glob

from platform import system
from os.path import sep

from helpers.write_json import write_json

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
        try:
            cur_date = split("-*-", split_timestamps_name)[3]
            cur_timestamp = split("-*-", split_timestamps_name)[4]
        except IndexError:  # The Synapse protocol name got changed at some point for some reason
            cur_date = split("-*-", split_timestamps_name)[2]
            cur_timestamp = split("-*-", split_timestamps_name)[3]
    else:
        print('Experiment type  not found in file name', flush=True)
        return None, None, None, None, None

    subject_id = split("_*_", recording_id)[0]
    subj_date = subject_id + '_' + cur_date

    # These are in alphabetical order. Must sort by date_trial or match with file
    key_path_info = glob(keys_path + sep + subject_id + '*' +
                         cur_date + '*' + cur_timestamp + "*_trialInfo.csv")

    if len(key_path_info) == 0:
        print("Key not found for " + recording_id, flush=True)
        return None, None, None, None, None

    # Load key and spout files
    info_key_times = read_csv(key_path_info[0])
    key_file_name = split(REGEX_SEP, key_path_info[0])[-1]

    if 'Aversive' in key_file_name:
        key_path_spout = glob(keys_path + sep + subject_id + '*' +
                              cur_date + '*' + cur_timestamp + "*spoutTimestamps.csv")
        spout_key_times = read_csv(key_path_spout[0])
        trial_types = ['Hit (all)', 'Hit (shock)', 'Hit (no shock)', 'Miss (shock)', 'Miss (no shock)', 'False alarm']
    elif 'Passive' in key_file_name:
        spout_key_times = None
        trial_types = ['Passive', ]
    elif '1IFC' in key_file_name:
        spout_key_times = None
        trial_types = ['Hit', 'Miss', 'Reject', 'False alarm']
    else:
        print('Experiment type not found in key file name', flush=True)
        return

    # If no JSON for that session exists, create sessionData
    analysis_id = settings_dict['ANALYSIS_ID']
    t_or_r_aligned = settings_dict['TRIAL_OR_RESPONSE_ALIGNED']
    try:
        # Load existing JSONs; will be empty if this is the first time running
        json_filenames = glob(settings_dict['OUTPUT_PATH'] + sep + 'JSON files' + sep + '*json')

        cur_sessionData_name = json_filenames[
            json_filenames.index(settings_dict['OUTPUT_PATH'] + sep + 'JSON files' + sep + subj_date + '_sessionData.json')]

        with open(cur_sessionData_name, 'r') as json_file:
            cur_sessionData = json.load(json_file)
    except ValueError:  # Create new sessionData
        cur_sessionData = {
            'Subject': subject_id,
            'Date': cur_date,
            'AnalysisID': {
                analysis_id: {
                    'Sessions': [],
                    'Alignment': {}
                }
            }
        }

    if analysis_id not in cur_sessionData['AnalysisID']:
        cur_sessionData['AnalysisID'].update({
            analysis_id: {
                'Sessions': [],
                'Alignment': {}
            }
        })

    session_id = key_file_name[:-4]
    if session_id not in cur_sessionData['AnalysisID'][analysis_id]['Sessions']:
        cur_sessionData['AnalysisID'][analysis_id]['Sessions'].append(session_id)

    if t_or_r_aligned not in cur_sessionData['AnalysisID'][analysis_id]['Alignment']:
        cur_sessionData['AnalysisID'][analysis_id]['Alignment'].update({t_or_r_aligned: {
            'Trial type': {}}})

    # Store full settings just in case
    cur_sessionData['AnalysisID'][analysis_id]['Alignment'][t_or_r_aligned]['Settings'] = settings_dict

    for trial_type in trial_types:
        cur_sessionData['AnalysisID'][analysis_id]['Alignment'][t_or_r_aligned]['Trial type'].update({trial_type: {}})


    if settings_dict['PIPELINE_SWITCHBOARD']['output_sessionData_json']:
        write_json(cur_sessionData, settings_dict['OUTPUT_PATH'] + sep + 'JSON files',
                   cur_sessionData['Subject'] + '_' + cur_sessionData['Date'] + '_sessionData.json')

    return subj_date, info_key_times, spout_key_times, trial_types, cur_sessionData