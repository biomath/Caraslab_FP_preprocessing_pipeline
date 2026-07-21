"""Core signal-extraction engine: slices per-trial dF/F windows, baseline z-scores them, and hands off to plotting/measurement.

Notes
-----
This module relies on ``np`` (numpy) and ``sep`` (``os.path.sep``) being
brought into its namespace via the ``from helpers.plotting_and_measurements_helpers
import *`` wildcard import below, rather than importing them directly itself.
"""
import platform

from copy import deepcopy

import pandas as pd

from scipy.signal import decimate
from scipy.stats import mode

from helpers.plotting_and_measurements_helpers import *
from helpers.preprocess_files import preprocess_files
from helpers.stopwatch import tic, toc


# Tweak the regex file separator for cross-platform compatibility
if platform.system() == 'Windows':
    REGEX_SEP = sep * 2
else:
    REGEX_SEP = sep


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
                             recording_fs=None):
    """Slice per-trial dF/F and baseline signal windows out of a continuous recording.

    Iterates over every trial row in ``key_times_df``, skips reminder
    trials, optionally filters by response latency, and computes the
    trial-window and baseline-window signal (aligned either to trial onset
    or to response time) for each remaining trial.

    Parameters
    ----------
    processed_signal : pandas.DataFrame
        Continuous recording with at least a ``'Time'`` column and either
        ``'Ch465_dff'`` or ``'Ch465_mV'`` (selected via ``subtract_405``).
    key_times_df : pandas.DataFrame
        Per-trial event/behavior table for the current trial type (column
        names are lower-cased in place on entry to normalize
        Aversive/Appetitive paradigm naming differences).
    signal_start_for_zscore : float, optional
        Start of the extraction window, in seconds relative to the
        alignment point. Default -2.0.
    signal_end_for_zscore : float, optional
        End of the extraction window, in seconds relative to the alignment
        point. Default 4.0.
    baseline_start_for_zscore : float, optional
        Start of the baseline window, in seconds relative to the alignment
        point (or to trial onset if ``use_nonAM_baseline`` is True).
        Default 0.0.
    baseline_end_for_zscore : float, optional
        End of the baseline window, in seconds relative to the alignment
        point (or to trial onset if ``use_nonAM_baseline`` is True).
        Default 1.0.
    response_latency_filter : float or bool, optional
        Minimum response latency (seconds) required to keep a trial; trials
        below this are skipped. ``False``/``0`` disables filtering. Default False.
    align_to_response : bool, optional
        If True, align the signal/baseline windows to response time instead
        of trial onset. Default False.
    use_nonAM_baseline : bool, optional
        If True, always take the baseline window relative to trial onset
        (before sound onset) even when ``align_to_response`` is True.
        Default False.
    subtract_405 : bool, optional
        If True, use the 405-fit-subtracted ``'Ch465_dff'`` column; if
        False, use the raw ``'Ch465_mV'`` column. Default True.
    ms_latency_values : bool, optional
        If True, response-latency columns are assumed to be in milliseconds
        and are divided by 1000 before use. Default False.
    recording_fs : float or None, optional
        Sampling rate (Hz) of ``processed_signal``, used to truncate
        signal/baseline snippets to their exact expected length (correcting
        for off-by-a-few-samples variation from downsampling). If ``None``,
        no length truncation is applied.

    Returns
    -------
    list
        One entry per retained trial:
        ``[[trial_id, amdepth_dB, trial_onset, trial_offset, resp_latency], dff_signal, baseline_signal]``
        where ``amdepth_dB`` is the AM depth converted to dB (or ``-40`` as
        a sentinel for zero/no AM depth), and ``dff_signal``/``baseline_signal``
        are 1-D numpy arrays of the extracted trial/baseline windows.
    """
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

            # Treat Miss (no shock) as special cases here.
            # For interesting comparison with response trials, align with the end of the AM if there is no latency value
            #   (i.e., potential Miss (attempt))
            if (np.isnan(response_time)) & (cur_trial['trialtype'] == 'Miss (no shock)'):
                response_time = 1
        else:
            response_time = 0

        _signal_start = (cur_trial['trial_onset'] + signal_start_for_zscore + response_time)
        _signal_end = (cur_trial['trial_onset'] + signal_end_for_zscore + response_time)
        signal_around_trial = processed_signal[
            (processed_signal['Time'] >= _signal_start) &
            (processed_signal['Time'] < _signal_end)]

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
            (processed_signal['Time'] >= _baseline_start) &
            (processed_signal['Time'] < _baseline_end)][
            sig_column].values

# # Truncate signal at response
        # if truncate_at_responseTime:
        #     response_time = cur_trial[cur_trial.keys().str.contains('resplatency')].iloc[0] / 1000
        #     signal_around_trial[signal_around_trial['Time'] >= (cur_trial['trial_onset'] + response_time)] = np.NaN
        #
        # baseline_mean = np.nanmean(baseline_signal)
        # baseline_std = np.nanstd(baseline_signal)

        dff_signal = signal_around_trial[sig_column].values

        # Truncate to exact expected length based on time window and sampling rate
        # This corrects for slight variations in snippet length due to downsampling
        if recording_fs is not None:
            expected_length = int((signal_end_for_zscore - signal_start_for_zscore) * recording_fs)
            expected_baseline_length = int((baseline_end_for_zscore - baseline_start_for_zscore) * recording_fs)
            # Just truncate if too long (avoids resampling artifacts)
            if len(dff_signal) > expected_length:
                dff_signal = dff_signal[:expected_length]
            if len(baseline_signal) > expected_baseline_length:
                baseline_signal = baseline_signal[:expected_baseline_length]

        # dff_zscore = (signal_around_trial[sig_column].values - baseline_mean) / baseline_std

        # Trial parameters will be under index 0, signal will always be index 1
        # Convert amdepth to dB and round
        cur_amdepth = cur_trial[cur_trial.keys().str.contains('amdepth')].iloc[0]
        if cur_amdepth > 0:
            cur_amdepth = np.round(20 * np.log10(cur_amdepth), 1)
        else:
            # 0% AM depth (no modulation) has no real dB value (log10(0) is undefined);
            # -40 dB is used as an arbitrary sentinel low enough to sort/plot before real depths
            cur_amdepth = -40
        ret_list.append([[cur_trial['trialid'],
                          cur_amdepth,
                          cur_trial['trial_onset'],
                          cur_trial['trial_offset'],
                          cur_trial[cur_trial.keys().str.contains('resplatency')].iloc[0]],
                         dff_signal, baseline_signal])
    return ret_list



def __get_trialID_zscore(trial_type_dict):
    """Z-score each trial's dF/F signal against its own baseline window.

    For every trial type and every trial within it, computes
    ``(dff_signal - baseline_mean) / baseline_std`` using
    ``ddof=1`` for the baseline standard deviation.

    Parameters
    ----------
    trial_type_dict : dict
        Mapping of trial-type name to a dict with (at least) ``'dff_signal'``
        (list of 1-D arrays) and ``'dff_baseline'`` (list of 1-D arrays),
        as produced by :func:`run_zscore_extraction`.

    Returns
    -------
    dict
        Deep copy of ``trial_type_dict`` with a ``'zscore'`` key added to
        each trial type's dict, containing the list of per-trial z-scored
        signals.

    Notes
    -----
    A trial baseline window with a single sample yields ``ddof=1`` standard
    deviation of ``NaN``, which silently propagates into that trial's
    z-score without a warning.
    """
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
    """Extract, z-score, and (per the switchboard) plot/measure trial-aligned photometry signals for a session batch.

    For each recording path: preprocesses/loads its key files, determines
    the behavioral paradigm (1IFC vs. AversiveAM), loads and optionally
    downsamples the continuous dF/F signal, then for every relevant trial
    type calls :func:`__get_trialID_dff_signal` to slice per-trial windows.
    After all recordings are processed, trial signals are truncated to a
    common length (excluding outlier-length trials beyond
    ``recording_fs / 2`` samples from the median), z-scored via
    :func:`__get_trialID_zscore`, and then — depending on
    ``SETTINGS_DICT['PIPELINE_SWITCHBOARD']`` flags — plotted and/or measured
    via functions from :mod:`helpers.plotting_and_measurements_helpers`.

    Parameters
    ----------
    input_list : tuple of (list of str, dict)
        Packed positional argument (single-argument convention for
        ``multiprocessing.Pool.map`` compatibility): ``(session_date_paths,
        SETTINGS_DICT)``.

        session_date_paths : list of str
            Recording paths for one (subject, date) unit of work.
        SETTINGS_DICT : dict
            Pipeline settings. Reads (among others) ``'SIGNAL_START_END'``,
            ``'BASELINE_START_END'``, ``'AUC_START_END'``, ``'SUBTRACT_405'``,
            ``'TARGET_ONSET_OFFSET'``, ``'ANALYSIS_ID'``, ``'OUTPUT_PATH'``,
            ``'OUTPUT_PLOTS_PATH'``, ``'RESPONSE_LATENCY_FILTER'``,
            ``'DOWNSAMPLE_RATE'``, ``'MS_LATENCY_VALUES'``,
            ``'USE_NONAM_BASELINE'``, ``'TRIAL_OR_RESPONSE_ALIGNED'``,
            ``'EXPERIMENT_TYPE'``, ``'SAMPLING_RATE'``, and
            ``'PIPELINE_SWITCHBOARD'`` (gates plotting/measurement/JSON output).

    Returns
    -------
    None
        Side effects only: prints progress, and (depending on the
        switchboard) writes plots, measurement CSVs, and/or the updated
        sessionData JSON to disk.

    Raises
    ------
    UserWarning
        If ``session_date_paths`` is empty.
    """
    (session_date_paths, settings_dict) = input_list

    # Load globals
    signal_start_end = settings_dict['SIGNAL_START_END']
    baseline_start_end = settings_dict['BASELINE_START_END']
    auc_start_end = settings_dict['AUC_START_END']

    subtract_405 = settings_dict['SUBTRACT_405']

    target_onset_offset = settings_dict['TARGET_ONSET_OFFSET']

    analysis_id = settings_dict['ANALYSIS_ID']

    output_path = settings_dict['OUTPUT_PATH']
    output_plots_path = settings_dict['OUTPUT_PLOTS_PATH']

    response_latency_filter = settings_dict['RESPONSE_LATENCY_FILTER']

    downsample_fs = settings_dict['DOWNSAMPLE_RATE']

    ms_latency_values = settings_dict['MS_LATENCY_VALUES']

    use_nonAM_baseline = settings_dict['USE_NONAM_BASELINE']

    # For data compactness when saving csvs and json files
    _precision_decimals = 3

    # Plot colors and some parameters
    passivePre_color = 'black'
    passivePost_color = 'darkgray'
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
    t_or_r_align = settings_dict['TRIAL_OR_RESPONSE_ALIGNED']

    if t_or_r_align == 'trial_aligned':
        align_to_response = False
    else:
        align_to_response = True

    trial_type_dict = dict()
    recording_fs = 0
    subj_date = ''
    t0 = tic()

    if len(session_date_paths) == 0:
        raise UserWarning('No key files were found. Please check your paths.')

    for recording_path in session_date_paths:
        subj_date, info_key_times, _, trial_types, cur_sessionData = preprocess_files(recording_path, settings_dict)

        if info_key_times is None:  # If preprocessing can't find files, skip
            print((recording_path + ': No info file was found\n'), end='', flush=True)
            continue

        # Reset this here, because passive files will change this to 0
        response_latency_filter = settings_dict['RESPONSE_LATENCY_FILTER']

        # Determine experiment type
        if '1ifc' in recording_path.lower() or '1ifc' in settings_dict['EXPERIMENT_TYPE'].lower():
            paradigm_type = '1IFC'
        elif ('aversive' in recording_path.lower() or 'passive' in recording_path.lower()
              or 'aversiveam' in settings_dict['EXPERIMENT_TYPE'].lower()):
            paradigm_type = 'AversiveAM'
        else:
            print('Behavioral experiment type not recognized... ', end='', flush=True)
            continue

        # Load signal
        processed_signal = pd.read_csv(recording_path)

        # If not set, approximate sampling rate
        if settings_dict['SAMPLING_RATE'] is None:
            # Find the average distance between points, which approximates TDT sampling rate
            recording_fs = 1 / np.mean(np.diff(processed_signal['Time']))
            if recording_fs == 0:
                print('Something weird with' + recording_path + 'sampling frequency estimation. Check key file.', end='', flush=True)
                continue
        else:
            recording_fs = settings_dict['SAMPLING_RATE']

        if downsample_fs is not None:
            downsample_q = int(recording_fs // downsample_fs)
            downsampled_data = {}
            for col in processed_signal.columns:
                if col == 'Time':
                    # Keep original time points (every Nth sample)
                    downsampled_data[col] = processed_signal[col].values[::downsample_q]
                else:
                    # Use zero_phase=True to avoid time shift in signal data
                    downsampled_data[col] = decimate(processed_signal[col].values, downsample_q, zero_phase=True)

            processed_signal = pd.DataFrame(downsampled_data)
            recording_fs = downsample_fs

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
                                                   recording_fs=recording_fs)

            trial_type_dict.update({trial_type: {'info': [x[0] for x in cur_signals],
                                                 'dff_signal': [x[1] for x in cur_signals],
                                                 'dff_baseline': [x[2] for x in cur_signals]}})
    toc(t0)

    # Uniformize lengths and exclude truncated signals by more than half sampling rate points
    # The median length should be the target
    tolerance = recording_fs / 2
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
    if settings_dict['PIPELINE_SWITCHBOARD']['plot_trial_zscores']:
        print('Plotting summary trial z-scores... ', end='', flush=True)
        t0 = tic()
        plot_FP_trial_zscore_summary(trial_type_dict, align_to_response, subj_date,
                                     output_plots_path, target_onset_offset[0], target_onset_offset[1],
                                     paradigm_type, response_latency_filter,
                                     signal_start_end[0], signal_start_end[1],
                                     ams_to_analyze, _precision_decimals,
                                     all_hit_color, hitShock_color, hitNoShock_color, miss_color, missShock_color,
                                     missNoShock_color, fa_color, reject_color, passivePre_color, passivePost_color)
        toc(t0)

    # Plot responses split by AM depth
    if settings_dict['PIPELINE_SWITCHBOARD']['plot_AMDepth_zscores']:
        print('Plotting z-scores by AM depth... ', end='', flush=True)
        t0 = tic()
        plot_FP_trial_zscore_byAMdepth(trial_type_dict, align_to_response, subj_date,
                                       output_plots_path, target_onset_offset[0], target_onset_offset[1],
                                       paradigm_type, response_latency_filter, signal_start_end[0], signal_start_end[1],
                                       ams_to_analyze, _precision_decimals)
        toc(t0)

    if settings_dict['PIPELINE_SWITCHBOARD']['extract_trial_zscores']:
        print('Extracting trial-by-trial z-scores... ', end='', flush=True)
        t0 = tic()
        output_sessionData_json = settings_dict['PIPELINE_SWITCHBOARD']['output_sessionData_json']
        measure_signals_and_save(trial_type_dict, cur_sessionData, analysis_id, t_or_r_align, subj_date,
                                 output_plots_path,
                                 baseline_start_end[0], baseline_start_end[1],
                                 signal_start_end[0], signal_start_end[1],
                                 auc_start_end[0], auc_start_end[1],
                                 min_length, ams_to_analyze, _precision_decimals,
                                 output_path, output_sessionData_json)
        toc(t0)


# ──────────────────────────────────────────────────────────────────────────────
# DRAFT CODE: extinction / pump-triggered spout-off pipeline.
# Not fully validated — see the "Notes" section of the docstring below for
# known bugs. Treat this function and everything it calls as a work in progress.
# ──────────────────────────────────────────────────────────────────────────────
def run_zscore_extraction_pumpTriggered_spoutOff(input_list):
    """Extract z-scored photometry signals triggered by spout offset during pump zero-flow windows.

    Intended for the Aversive-AM "extinction" paradigm: for each recording,
    locates windows where ``PumpRate`` is 0 across consecutive trials, finds
    the first spout-offset event within each such window, and extracts a
    dF/F/baseline signal snippet aligned to that spout-offset time (analogous
    to :func:`run_zscore_extraction`, but triggered by pump/spout events
    rather than trial onset/response). Follows with the same
    length-uniformization, z-scoring, and (switchboard-gated)
    plotting/measurement steps as :func:`run_zscore_extraction`.

    Parameters
    ----------
    input_list : tuple of (list of str, dict)
        Packed positional argument: ``(session_date_paths, SETTINGS_DICT)``.
        See :func:`run_zscore_extraction` for the shared settings keys; this
        function additionally requires ``info_key_times`` to contain a
        ``'PumpRate'`` column and a corresponding ``spoutTimestamps.csv`` to
        exist for the recording (via ``preprocess_files``).

    Returns
    -------
    None
        Side effects only, mirroring :func:`run_zscore_extraction`.

    Raises
    ------
    UserWarning
        If ``session_date_paths`` is empty.

    Notes
    -----
    This function is now called from
    :func:`helpers.run_FP_pipeline.run_pipeline`'s extinction/pump-triggered
    switchboard branch, but it still contains unresolved latent bugs:
    ``info_key_times[trial_idx, 'Trial_onset']``-style tuple indexing on a
    DataFrame is invalid pandas usage, and ``relevant_spoutOffset =
    spout_offsets[mask][0]`` reduces to a scalar before the subsequent
    ``len(relevant_spoutOffset) == 0`` check, which would raise
    ``TypeError`` on a numpy scalar. Its downstream call to
    :func:`~helpers.plotting_and_measurements_helpers.measure_extinction_spoutOffset_signals_and_save`
    also passes one fewer positional argument than that function requires.
    These have not been fixed.
    """
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

    ###### PLOTTING AND MEASUREMENTS START HERE ######
    # DRAFT CODE: see the module-level draft-code note above run_zscore_extraction_pumpTriggered_spoutOff.
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