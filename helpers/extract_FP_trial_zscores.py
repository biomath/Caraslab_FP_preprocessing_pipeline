import platform
from datetime import datetime
from copy import deepcopy
from os.path import sep
import csv
import json

import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from matplotlib import patches, colormaps
from scipy.integrate import simpson
from scipy.signal import resample, decimate
from scipy.stats import mode

from helpers.format_axes import format_ax
from helpers.plotting_and_measurements_helpers import *
from helpers.preprocess_files import preprocess_files
from helpers.stopwatch import tic, toc
from helpers.write_json import write_json

# Tweak the regex file separator for cross-platform compatibility
if platform.system() == 'Windows':
    REGEX_SEP = sep * 2
else:
    REGEX_SEP = sep


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def __get_trialID_dff_signal(processed_signal, key_times_df,
                             signal_start_for_zscore=-2.,
                             signal_end_for_zscore=4.,
                             baseline_start_for_zscore=0.,
                             baseline_end_for_zscore=1.,
                             response_latency_filter=False,
                             align_to_response=False,
                             use_nonAM_baseline=False,
                             subtract_405=True,
                             ms_latency_values=False,
                             sampling_frequency=None,
                             downsample_fs=None):
    ret_list = list()
    # There are inconsistencies between the Aversive and Appetitive paradigm in the column naming.
    # Make all lower case here
    key_times_df.columns = key_times_df.columns.str.lower()
    for _, cur_trial in key_times_df.iterrows():
        # remove reminders
        if cur_trial[cur_trial.keys().str.contains('reminder')].iloc[0] == 1:
            continue

        # Use response time to get at the reward delivery
        if ms_latency_values:
            cur_trial.loc[cur_trial.keys().str.contains('resplatency')] /= 1000

        # Only include trials with higher than a specified response time
        # If response_latency_filter == 0, the first if-statement is redundant, but implement it just in case
        if response_latency_filter > 0:
            if cur_trial[cur_trial.keys().str.contains('resplatency')].iloc[0] < response_latency_filter:
                continue

        if align_to_response:
            response_time = cur_trial[cur_trial.keys().str.contains('resplatency')].iloc[0]
        else:
            response_time = 0

        _signal_start = (cur_trial['trial_onset'] + signal_start_for_zscore + response_time)
        _signal_end = (cur_trial['trial_onset'] + signal_end_for_zscore + response_time)
        signal_around_trial = processed_signal[
            (processed_signal['Time'] > _signal_start) &
            (processed_signal['Time'] <= _signal_end)]

        # 405 fit-removed signal
        if subtract_405:
            sig_column = 'Ch465_dff'
        else:
            sig_column = 'Ch465_mV'

        # z-score it
        if use_nonAM_baseline:
            # Use baseline before sound onset (even when aligned by response time)
            _baseline_start = (cur_trial['trial_onset'] + baseline_start_for_zscore)
            _baseline_end = (cur_trial['trial_onset'] + baseline_end_for_zscore)
        else:
            # Use baseline just before response
            _baseline_start = (cur_trial['trial_onset'] + baseline_start_for_zscore + response_time)
            _baseline_end = (cur_trial['trial_onset'] + baseline_end_for_zscore + response_time)

        baseline_signal = processed_signal[
            (processed_signal['Time'] > _baseline_start) &
            (processed_signal['Time'] <= _baseline_end)][
            sig_column].values

# # Truncate signal at response
        # if truncate_at_responseTime:
        #     response_time = cur_trial[cur_trial.keys().str.contains('resplatency')].iloc[0] / 1000
        #     signal_around_trial[signal_around_trial['Time'] >= (cur_trial['trial_onset'] + response_time)] = np.NaN
        #
        # baseline_mean = np.nanmean(baseline_signal)
        # baseline_std = np.nanstd(baseline_signal)

        dff_signal = signal_around_trial[sig_column].values

        if downsample_fs is not None:
            downsample_q = int(sampling_frequency // downsample_fs)
            try:
                dff_signal = decimate(dff_signal, downsample_q)
            except ValueError:
                # If signals are too short this will fail. Remove signal
                continue

            try:
                baseline_signal = decimate(baseline_signal, downsample_q)
            except ValueError:
                # If signals are too short this will fail. Remove signal
                continue

        # dff_zscore = (signal_around_trial[sig_column].values - baseline_mean) / baseline_std

        # Trial parameters will be under index 0, signal will always be index 1
        # Convert amdepth to dB and round
        cur_amdepth = cur_trial[cur_trial.keys().str.contains('amdepth')].iloc[0]
        if cur_amdepth > 0:
            cur_amdepth = np.round(20 * np.log10(cur_amdepth), 1)
        else:
            cur_amdepth = -40
        ret_list.append([[cur_trial['trialid'],
                          cur_amdepth,
                          cur_trial['trial_onset'],
                          cur_trial['trial_offset'],
                          cur_trial[cur_trial.keys().str.contains('resplatency')].iloc[0]],
                         dff_signal, baseline_signal])
    return ret_list



def __get_trialID_zscore(trial_type_dict):
    ret_dict = deepcopy(trial_type_dict)
    for trial_type_key in trial_type_dict.keys():
        allTrial_dff_signal = trial_type_dict[trial_type_key]['dff_signal']
        allTrial_dff_baseline = trial_type_dict[trial_type_key]['dff_baseline']

        dff_zscore_list = list()
        for trial_idx, trial_signal in enumerate(allTrial_dff_signal):
            trial_baseline = allTrial_dff_baseline[trial_idx]

            # z-score it
            baseline_mean = np.nanmean(trial_baseline)
            baseline_std = np.nanstd(trial_baseline, ddof=1)

            dff_zscore = (trial_signal - baseline_mean) / baseline_std
            dff_zscore_list.append(dff_zscore)

        ret_dict[trial_type_key].update({'zscore': dff_zscore_list})

    return ret_dict


def run_zscore_extraction(input_list):
    (session_date_paths, SETTINGS_DICT) = input_list

    # Load globals
    signal_start_end = SETTINGS_DICT['SIGNAL_START_END']
    baseline_start_end = SETTINGS_DICT['BASELINE_START_END']
    auc_start_end = SETTINGS_DICT['AUC_START_END']

    subtract_405 = SETTINGS_DICT['SUBTRACT_405']

    target_onset_offset = SETTINGS_DICT['TARGET_ONSET_OFFSET']

    analysis_id = SETTINGS_DICT['ANALYSIS_ID']

    output_path = SETTINGS_DICT['OUTPUT_PATH']
    output_plots_path = SETTINGS_DICT['OUTPUT_PLOTS_PATH']

    response_latency_filter = SETTINGS_DICT['RESPONSE_LATENCY_FILTER']

    downsample_fs = SETTINGS_DICT['DOWNSAMPLE_RATE']

    ms_latency_values = SETTINGS_DICT['MS_LATENCY_VALUES']

    use_nonAM_baseline = SETTINGS_DICT['USE_NONAM_BASELINE']
    # For data compactness when saving csvs and json files
    _precision_decimals = 3

    # Plot colors and some parameters
    passive_color = 'black'
    all_hit_color = '#0F52BA'
    hit_color = '#60B2E5'
    fa_color = '#F0A202'
    miss_color = '#C84630'

    # Specific to Aversive paradigm
    hitShock_color = hit_color
    hitNoShock_color = hit_color
    missShock_color = miss_color
    missNoShock_color = miss_color

    # Specific to 1IFC
    reject_color = '#2B4570'

    # Trial-aligned: aligned to trial onset
    # Response-aligned: aligned to spout offset triggering outcome (aversive) OR reward trigger (1IFC)
    t_or_r_align = SETTINGS_DICT['TRIAL_OR_RESPONSE_ALIGNED']

    if t_or_r_align == 'trial_aligned':
        align_to_response = False
    else:
        align_to_response = True

    trial_type_dict = dict()
    sampling_frequency = 0
    subj_date = ''
    print('Loading and z-scoring signals... ', end='', flush=True)
    t0 = tic()

    if len(session_date_paths) == 0:
        raise UserWarning('No key files were found. Please check your paths.')

    for recording_path in session_date_paths:
        subj_date, info_key_times, _, trial_types, cur_sessionData = preprocess_files(recording_path, SETTINGS_DICT)
        if info_key_times is None:  # If preprocessing can't find files, skip
            continue

        # Reset this here, because passive files will change this to 0
        response_latency_filter = SETTINGS_DICT['RESPONSE_LATENCY_FILTER']

        # Determine experiment type
        if '1IFC' in recording_path or '1IFC' in SETTINGS_DICT['EXPERIMENT_TYPE']:
            paradigm_type = '1IFC'
        elif ('Aversive' in recording_path or 'Passive' in recording_path
              or 'AversiveAM' in SETTINGS_DICT['EXPERIMENT_TYPE']):
            paradigm_type = 'AversiveAM'
        else:
            print('Behavioral experiment type not recognized... ', end='', flush=True)
            continue

        # Load signal
        processed_signal = pd.read_csv(recording_path)

        # If not set, approximate sampling rate
        if SETTINGS_DICT['SAMPLING_RATE'] is None:
            # Find the most common difference between time points
            # This is more robust than the mean since sessions can contain gaps in time due to artifact removal
            sampling_frequency = 1 / mode(np.diff(processed_signal['Time']))[0]
            if sampling_frequency == 0:
                print('Something weird with' + recording_path + 'sampling frequency estimation. Check key file.', end='', flush=True)
                continue
        else:
            sampling_frequency = SETTINGS_DICT['SAMPLING_RATE']

        for trial_type in trial_types:
            cur_trial_align_to_response = align_to_response  # Will be changed temporarily for passive sessions
            if paradigm_type == '1IFC':
                if trial_type == 'Hit':
                    cur_key_times = info_key_times[(info_key_times['Hit'] == 1)]
                elif trial_type == 'Miss':
                    cur_key_times = info_key_times[(info_key_times['Miss'] == 1)]
                elif trial_type == 'Reject':
                    cur_key_times = info_key_times[(info_key_times['CR'] == 1)]
                elif trial_type == 'False alarm':
                    cur_key_times = info_key_times[(info_key_times['FA'] == 1)]
                else:  # Passive
                    # Sometimes responses can be recorded in passive sessions if the spout override is interrupted during a trial
                    # Ensure trial onset is used always by turning this flag off
                    cur_trial_align_to_response = False
                    cur_key_times = info_key_times[(info_key_times['TrialType'] == 0)]
                    # Keep track of trial number and onset time too
            elif paradigm_type == 'AversiveAM':
                if trial_type == 'Hit (all)':
                    cur_key_times = info_key_times[
                        (info_key_times['TrialType'] == 0) & (info_key_times['Hit'] == 1) & (
                                info_key_times['Reminder'] == 0)]

                elif trial_type == 'Hit (shock)':
                    cur_key_times = info_key_times[
                        (info_key_times['TrialType'] == 0) & (info_key_times['Hit'] == 1) & (
                                info_key_times['Reminder'] == 0) &
                        (info_key_times['ShockFlag'] == 1)]

                elif trial_type == 'Hit (no shock)':
                    cur_key_times = info_key_times[
                        (info_key_times['TrialType'] == 0) & (info_key_times['Hit'] == 1) & (
                                info_key_times['Reminder'] == 0) &
                        (info_key_times['ShockFlag'] == 0)]

                elif trial_type == 'Miss (shock)':
                    cur_key_times = info_key_times[
                        (info_key_times['TrialType'] == 0) & (info_key_times['Miss'] == 1) & (
                                info_key_times['Reminder'] == 0) &
                        (info_key_times['ShockFlag'] == 1)]

                elif trial_type == 'Miss (no shock)':
                    cur_key_times = info_key_times[
                        (info_key_times['TrialType'] == 0) & (info_key_times['Miss'] == 1) & (
                                info_key_times['Reminder'] == 0) &
                        (info_key_times['ShockFlag'] == 0)]

                elif trial_type == 'False alarm':
                    cur_key_times = info_key_times[
                        (info_key_times['FA'] == 1) & (info_key_times['Reminder'] == 0)]

                else:  # Passive
                    cur_key_times = info_key_times[(info_key_times['TrialType'] == 0)]
                    response_latency_filter = 0
                    # Sometimes responses can be recorded in passive sessions if the spout override is interrupted during a trial
                    # Ensure trial onset is used always by turning this flag off
                    cur_trial_align_to_response = False
            else:
                print('Experiment type not recognized. Exiting.', flush=True)
                return

            cur_signals = __get_trialID_dff_signal(processed_signal, cur_key_times,
                                                   signal_start_for_zscore=signal_start_end[0],
                                                   signal_end_for_zscore=signal_start_end[1],
                                                   baseline_start_for_zscore=baseline_start_end[0],
                                                   baseline_end_for_zscore=baseline_start_end[1],
                                                   response_latency_filter=response_latency_filter,
                                                   align_to_response=cur_trial_align_to_response,
                                                   use_nonAM_baseline=use_nonAM_baseline,
                                                   subtract_405=subtract_405,
                                                   ms_latency_values=ms_latency_values,
                                                   sampling_frequency=sampling_frequency,
                                                   downsample_fs=downsample_fs)

            trial_type_dict.update({trial_type: {'info': [x[0] for x in cur_signals],
                                                 'dff_signal': [x[1] for x in cur_signals],
                                                 'dff_baseline': [x[2] for x in cur_signals]}})
    toc(t0)

    # Update sampling frequency to match the downsampled frequency
    if downsample_fs is not None and sampling_frequency > downsample_fs:
        downsample_q = int(sampling_frequency // downsample_fs)
        try:
            sampling_frequency /= downsample_q
        except ZeroDivisionError:
            print('Something weird with sampling frequency estimation. Check key files', end='',
                  flush=True)
            return


    # Uniformize lengths and exclude truncated signals by more than half sampling rate points
    # The median length should be the target
    tolerance = sampling_frequency / 2
    sig_lengths = []
    print('Uniformizing signal lengths for plotting... ', end='', flush=True)
    t0 = tic()
    for trial_type_key in trial_type_dict.keys():
        sig_lengths.extend([len(x) for x in trial_type_dict[trial_type_key]['dff_signal']])
    median_length = np.median(sig_lengths)

    sig_lengths = []
    for trial_type_key in trial_type_dict.keys():
        good_indices: list[int] = [idx for
                                   idx, x in enumerate(trial_type_dict[trial_type_key]['dff_signal']) if
                                   (len(x) > (median_length - tolerance)) and
                                   (len(x) < (median_length + 100))]

        trial_type_dict[trial_type_key]['info'] = [trial_type_dict[trial_type_key]['info'][x] for x in good_indices]
        trial_type_dict[trial_type_key]['dff_signal'] = [trial_type_dict[trial_type_key]['dff_signal'][x] for x in
                                                         good_indices]
        trial_type_dict[trial_type_key]['dff_baseline'] = [trial_type_dict[trial_type_key]['dff_baseline'][x] for x in
                                                           good_indices]

        sig_lengths.extend([len(x) for x in trial_type_dict[trial_type_key]['dff_signal']])

    # Check if there are any signals present
    if len(sig_lengths) == 0:
        print('No signals found. Tip: did you set the response latency filter properly?', flush=True)
        return

    # Now uniformize lengths
    min_length = np.min(sig_lengths)
    for trial_type_key in trial_type_dict.keys():
        trial_type_dict[trial_type_key]['dff_signal'] = [np.array(x[0:int(min_length)]) for
                                                         x in trial_type_dict[trial_type_key]['dff_signal']]

    toc(t0)

    # Extract Z-score
    trial_type_dict = __get_trialID_zscore(trial_type_dict)

    # Select specific AMs
    ams_to_analyze = None  # or None for all

    ###### PLOTTING AND MEASUREMENTS START HERE ######
    if SETTINGS_DICT['PIPELINE_SWITCHBOARD']['plot_trial_zscores']:
        print('Plotting summary trial z-scores... ', end='', flush=True)
        t0 = tic()
        plot_FP_trial_zscore_summary(trial_type_dict, align_to_response, subj_date,
                                     output_plots_path, target_onset_offset[0], target_onset_offset[1],
                                     paradigm_type, response_latency_filter,
                                     signal_start_end[0], signal_start_end[1],
                                     ams_to_analyze, _precision_decimals,
                                     all_hit_color, hitShock_color, hitNoShock_color, miss_color, missShock_color,
                                     missNoShock_color, fa_color, reject_color, passive_color)
        toc(t0)

    # Plot responses split by AM depth
    if SETTINGS_DICT['PIPELINE_SWITCHBOARD']['plot_AMDepth_zscores']:
        print('Plotting z-scores by AM depth... ', end='', flush=True)
        t0 = tic()
        plot_FP_trial_zscore_byAMdepth(trial_type_dict, align_to_response, subj_date,
                                       output_plots_path, target_onset_offset[0], target_onset_offset[1],
                                       paradigm_type, response_latency_filter, signal_start_end[0], signal_start_end[1],
                                       ams_to_analyze, _precision_decimals)
        toc(t0)

    if SETTINGS_DICT['PIPELINE_SWITCHBOARD']['extract_trial_zscores']:
        print('Extracting trial-by-trial z-scores... ', end='', flush=True)
        t0 = tic()
        output_sessionData_json = SETTINGS_DICT['PIPELINE_SWITCHBOARD']['output_sessionData_json']
        measure_signals_and_save(trial_type_dict, cur_sessionData, analysis_id, t_or_r_align, subj_date,
                                 output_plots_path,
                                 baseline_start_end[0], baseline_start_end[1],
                                 signal_start_end[0], signal_start_end[1],
                                 auc_start_end[0], auc_start_end[1],
                                 min_length, ams_to_analyze, _precision_decimals,
                                 sampling_frequency, output_path, output_sessionData_json)
        toc(t0)


def run_zscore_extraction_pumpTriggered_spoutOff(input_list):
    (session_date_paths, SETTINGS_DICT) = input_list

    # Load globals
    signal_start_end = SETTINGS_DICT['SIGNAL_START_END']
    baseline_start_end = SETTINGS_DICT['BASELINE_START_END']
    auc_start_end = SETTINGS_DICT['AUC_START_END']

    subtract_405 = SETTINGS_DICT['SUBTRACT_405']

    analysis_id = SETTINGS_DICT['ANALYSIS_ID']

    output_path = SETTINGS_DICT['OUTPUT_PATH']

    output_plots_path = SETTINGS_DICT['OUTPUT_PLOTS_PATH']

    downsample_fs = SETTINGS_DICT['DOWNSAMPLE_RATE']

    # For data compactness when saving csvs and json files
    _precision_decimals = 3

    # Plot colors and some parameters
    cur_color = 'black'

    trial_type_dict = dict()
    sampling_frequency = 0
    subj_date = ''
    print('Loading and z-scoring signals... ', end='', flush=True)
    t0 = tic()

    if len(session_date_paths) == 0:
        raise UserWarning('No key files were found. Please check your paths.')

    for recording_path in session_date_paths:
        subj_date, info_key_times, spout_key_times, trial_types, cur_sessionData = preprocess_files(recording_path, SETTINGS_DICT)
        if info_key_times is None or spout_key_times is None:  # If preprocessing can't find files, skip
            continue

        # Determine experiment type
        if not ('Aversive' in recording_path or 'Passive' in recording_path
              or 'AversiveAM' in SETTINGS_DICT['EXPERIMENT_TYPE']):
            print('This function is only relevant for Aversive AM task. Aborting...', end='', flush=True)
            return

        # Load signal
        processed_signal = pd.read_csv(recording_path)

        # If not set, approximate sampling rate
        if SETTINGS_DICT['SAMPLING_RATE'] is None:
            # Find the most common difference between time points
            # This is more robust than the mean since sessions can contain gaps in time due to artifact removal
            sampling_frequency = 1 / mode(np.diff(processed_signal['Time']))[0]
            if sampling_frequency == 0:
                print('Something weird with' + recording_path + 'sampling frequency estimation. Check key file.',
                      end='', flush=True)
                continue
        else:
            sampling_frequency = SETTINGS_DICT['SAMPLING_RATE']

        # Find first spout offset during a pump 0-flow window
        pump_rate_values = info_key_times['PumpRate'].values
        ## Find switches to 0 flow first
        zero_rate_onset_offset = []
        trial_idx = 0
        while trial_idx < len(pump_rate_values) - 1:
            pump_rate = pump_rate_values[trial_idx]
            if pump_rate == 0:
                # Zero flow block started
                start_onset = info_key_times[trial_idx, 'Trial_onset'].values
                next_trial_idx = trial_idx + 1
                while next_trial_idx < len(pump_rate_values):
                    next_pump_rate = pump_rate_values[next_trial_idx]
                    if next_trial_idx == (len(pump_rate_values) - 1):  # Last trial was 0-flow
                        end_offset = info_key_times[next_trial_idx, 'Trial_offset'].values
                        zero_rate_onset_offset.append((start_onset, end_offset))
                        break
                    elif next_pump_rate == 0:  # Next trial is still 0-flow
                        next_trial_idx += 1
                        continue
                    else:  # 0-flow block ended
                        end_offset = info_key_times[next_trial_idx-1, 'Trial_offset'].values
                        zero_rate_onset_offset.append((start_onset, end_offset))
                        trial_idx = next_trial_idx
                        break
            else:
                trial_idx += 1
                continue

        ## Grab spout offset triggered signals within windows
        cur_signals = []
        trial_id = []  # Assign an ID to each event
        spout_offsets = spout_key_times['Spout_offset'].values
        signal_start_for_zscore = signal_start_end[0]
        signal_end_for_zscore = signal_start_end[1]
        baseline_start_for_zscore = baseline_start_end[0]
        baseline_end_for_zscore = baseline_start_end[1]
        for trial_idx, onset_offset in enumerate(zero_rate_onset_offset):
            relevant_spoutOffset_mask = (spout_offsets > onset_offset[0]) & (spout_offsets < onset_offset[1])
            relevant_spoutOffset = spout_offsets[relevant_spoutOffset_mask][0]  # Grab the first one
            if len(relevant_spoutOffset) == 0:  # No spout offsets were found in window (unlikely)
                continue
            response_latency = relevant_spoutOffset - onset_offset[0]
            _signal_start = (relevant_spoutOffset + signal_start_for_zscore)
            _signal_end = (relevant_spoutOffset + signal_end_for_zscore)
            signal_around_trial = processed_signal[
                (processed_signal['Time'] > _signal_start) &
                (processed_signal['Time'] <= _signal_end)]

            # 405 fit-removed signal
            if subtract_405:
                sig_column = 'Ch465_dff'
            else:
                sig_column = 'Ch465_mV'

            # z-score it
            # Use baseline just before response
            _baseline_start = (relevant_spoutOffset + baseline_start_for_zscore)
            _baseline_end = (relevant_spoutOffset + baseline_end_for_zscore)

            baseline_signal = processed_signal[
                (processed_signal['Time'] > _baseline_start) &
                (processed_signal['Time'] <= _baseline_end)][
                sig_column].values

            dff_signal = signal_around_trial[sig_column].values

            if downsample_fs is not None:
                downsample_q = int(sampling_frequency // downsample_fs)
                try:
                    dff_signal = decimate(dff_signal, downsample_q)
                except ValueError:
                    # If signals are too short this will fail. Remove signal
                    continue

                try:
                    baseline_signal = decimate(baseline_signal, downsample_q)
                except ValueError:
                    # If signals are too short this will fail. Remove signal
                    continue

            cur_signals.append([
                (trial_idx+1, onset_offset[0],onset_offset[1], response_latency), dff_signal, baseline_signal
            ])

        trial_type_dict.update({'Zero_flow': {'info': [x[0] for x in cur_signals],
                                             'dff_signal': [x[1] for x in cur_signals],
                                             'dff_baseline': [x[2] for x in cur_signals]}})
    toc(t0)

    # Update sampling frequency to match the downsampled frequency
    if downsample_fs is not None and sampling_frequency > downsample_fs:
        downsample_q = int(sampling_frequency // downsample_fs)
        try:
            sampling_frequency /= downsample_q
        except ZeroDivisionError:
            print('Something weird with sampling frequency estimation. Check key files', end='',
                  flush=True)
            return

    # Uniformize lengths and exclude truncated signals by more than half sampling rate points
    # The median length should be the target
    tolerance = sampling_frequency / 2
    sig_lengths = []
    print('Uniformizing signal lengths for plotting... ', end='', flush=True)
    t0 = tic()
    for trial_type_key in trial_type_dict.keys():
        sig_lengths.extend([len(x) for x in trial_type_dict[trial_type_key]['dff_signal']])
    median_length = np.median(sig_lengths)

    sig_lengths = []
    for trial_type_key in trial_type_dict.keys():
        good_indices: list[int] = [idx for
                                   idx, x in enumerate(trial_type_dict[trial_type_key]['dff_signal']) if
                                   (len(x) > (median_length - tolerance)) and
                                   (len(x) < (median_length + 100))]

        trial_type_dict[trial_type_key]['info'] = [trial_type_dict[trial_type_key]['info'][x] for x in good_indices]
        trial_type_dict[trial_type_key]['dff_signal'] = [trial_type_dict[trial_type_key]['dff_signal'][x] for x in
                                                         good_indices]
        trial_type_dict[trial_type_key]['dff_baseline'] = [trial_type_dict[trial_type_key]['dff_baseline'][x] for x in
                                                           good_indices]

        sig_lengths.extend([len(x) for x in trial_type_dict[trial_type_key]['dff_signal']])

    # Check if there are any signals present
    if len(sig_lengths) == 0:
        print('No signals found. Tip: did you set the response latency filter properly?', flush=True)
        return

    # Now uniformize lengths
    min_length = np.min(sig_lengths)
    for trial_type_key in trial_type_dict.keys():
        trial_type_dict[trial_type_key]['dff_signal'] = [np.array(x[0:int(min_length)]) for
                                                         x in trial_type_dict[trial_type_key]['dff_signal']]

    toc(t0)

    # Extract Z-score
    trial_type_dict = __get_trialID_zscore(trial_type_dict)

    # Select specific AMs
    ams_to_analyze = None  # or None for all

    ###### PLOTTING AND MEASUREMENTS START HERE ######
    # Plot responses triggered by turning off the pump (0 mL/min flag)
    if SETTINGS_DICT['PIPELINE_SWITCHBOARD']['plot_extinction_spoutOff_zscores']:
        print('Plotting z-scores by AM depth... ', end='', flush=True)
        t0 = tic()
        plot_FP_extinction_spoutOff_zscores(trial_type_dict, subj_date,
                                            output_plots_path, signal_start_end[0],
                                            signal_start_end[1], _precision_decimals)
        toc(t0)

        if SETTINGS_DICT['PIPELINE_SWITCHBOARD']['extract_extinction_spoutOff_zscores']:
            print('Extracting 0 mL/min-triggered spout offset  z-scores... ', end='', flush=True)
            t0 = tic()
            output_sessionData_json = SETTINGS_DICT['PIPELINE_SWITCHBOARD']['output_sessionData_json']
            measure_extinction_spoutOffset_signals_and_save(
                trial_type_dict, cur_sessionData, analysis_id, subj_date,
                output_plots_path,
                baseline_start_end[0], baseline_start_end[1],
                signal_start_end[0], signal_start_end[1],
                auc_start_end[0], auc_start_end[1],
                min_length, _precision_decimals,
                output_path, output_sessionData_json)

            toc(t0)