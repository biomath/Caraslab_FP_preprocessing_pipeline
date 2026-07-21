"""Locates key files for a recording and creates/updates the per-session sessionData JSON scaffold."""
from re import split
from glob import glob

from platform import system
from os.path import sep

from helpers.write_json import write_json
from helpers.date_utils import find_date_in_tag, find_timestamp_in_tag

# Tweak the regex file separator for cross-platform compatibility
if system() == 'Windows':
    REGEX_SEP = sep * 2
else:
    REGEX_SEP = sep
import json

from pandas import read_csv


def load_session_files(memory_path, settings_dict):
    """Locate and load the trialInfo/spoutTimestamps key files for one recording.

    Parses the subject ID out of ``memory_path``'s filename, and the date/
    timestamp via :func:`~helpers.date_utils.find_date_in_tag` /
    :func:`~helpers.date_utils.find_timestamp_in_tag` (position-independent,
    based on the standalone 6-digit YYMMDD/HHMMSS tokens), then reads the
    matching ``*_trialInfo.csv`` (and, for aversive/extinction sessions,
    ``*_spoutTimestamps.csv``) from ``settings_dict['KEYS_PATH']``. Has no
    dependency on alignment or analysis JSON bookkeeping.

    Parameters
    ----------
    memory_path : str
        Path to the recording's signal file; only its filename is used to
        derive the subject ID, date, and timestamp.
    settings_dict : dict
        Pipeline settings dict. Must contain ``'KEYS_PATH'`` (directory to
        search for key files) and ``'EXPERIMENT_TYPE'`` (used as a fallback
        keyword when the filename itself doesn't disambiguate the layout).

    Returns
    -------
    subj_date : str or None
        ``'{subject_id}_{date}'`` identifier, or ``None`` if the experiment
        type/key file could not be resolved from the filename.
    info_key_times : pandas.DataFrame or None
        Contents of the matched ``*_trialInfo.csv`` file.
    spout_key_times : pandas.DataFrame or None
        Contents of the matched ``*_spoutTimestamps.csv`` file for
        aversive/extinction sessions; ``None`` for passive/1IFC sessions
        (which have no spout file) or if resolution failed.
    trial_types : list of str or None
        Trial-type labels expected for this session's paradigm (e.g.
        ``['Hit (all)', 'Hit (shock)', ...]`` for aversive/extinction).
    key_file_name : str or None
        Filename (no directory) of the matched ``*_trialInfo.csv``.

    Notes
    -----
    Returns the 5-tuple ``(None, None, None, None, None)`` whenever the
    experiment type can't be determined from the filename, or no matching
    trialInfo file is found on disk — callers must check for this sentinel.
    """
    keys_path = settings_dict['KEYS_PATH']

    split_memory_path = split(REGEX_SEP, memory_path)
    recording_id = split_memory_path[-1][:-4]  # strip the fixed-length ".csv" extension (4 chars)

    split_timestamps_name = split("_*_", recording_id)[1]
    is_known_paradigm = (
        '1ifc' in split_timestamps_name.lower() or '1ifc' in settings_dict['EXPERIMENT_TYPE'].lower() or
        'aversive' in split_timestamps_name.lower() or
        'passive' in split_timestamps_name.lower() or
        'extinction' in split_timestamps_name.lower()
    )
    if not is_known_paradigm:
        print('Experiment type not found in file name', flush=True)
        return None, None, None, None, None

    # Locate the date/timestamp by content (a standalone 6-digit token) rather than by a
    # fixed position, since the number of words preceding them varies across paradigms.
    cur_date = find_date_in_tag(split_timestamps_name)
    cur_timestamp = find_timestamp_in_tag(split_timestamps_name)

    subject_id = split("_*_", recording_id)[0]
    subj_date = subject_id + '_' + cur_date

    key_path_info = glob(keys_path + sep + subject_id + '*' +
                         cur_date + '*' + cur_timestamp + "*_trialInfo.csv")
    if len(key_path_info) == 0:
        print("Key not found for " + recording_id, flush=True)
        return None, None, None, None, None

    info_key_times = read_csv(key_path_info[0])
    key_file_name = split(REGEX_SEP, key_path_info[0])[-1]

    if 'aversive' in key_file_name.lower() or 'extinction' in key_file_name.lower():
        key_path_spout = glob(keys_path + sep + subject_id + '*' +
                              cur_date + '*' + cur_timestamp + "*spoutTimestamps.csv")
        spout_key_times = read_csv(key_path_spout[0])
        trial_types = ['Hit (all)', 'Hit (shock)', 'Hit (no shock)', 'Miss (shock)', 'Miss (no shock)', 'False alarm']
    elif 'passive' in key_file_name.lower():
        spout_key_times = None
        if 'pre' in key_file_name.lower():
            trial_types = ['Passive-Pre']
        elif 'post' in key_file_name.lower():
            trial_types = ['Passive-Post']
        else:
            trial_types = ['Passive']
    elif '1ifc' in key_file_name.lower():
        spout_key_times = None
        trial_types = ['Hit', 'Miss', 'Reject', 'False alarm']
    else:
        print('Experiment type not found in key file name', flush=True)
        return None, None, None, None, None

    return subj_date, info_key_times, spout_key_times, trial_types, key_file_name


def preprocess_files(memory_path, settings_dict):
    """Load a recording's key files and create/update its sessionData JSON scaffold.

    Calls :func:`load_session_files` to locate/parse the recording's key
    files, then loads the existing ``{subj_date}_sessionData.json`` (or
    initializes a new one) and records this analysis ID, session, alignment,
    and trial-type entries into it. Optionally writes the updated JSON back
    to disk depending on the pipeline switchboard.

    Parameters
    ----------
    memory_path : str
        Path to the recording's signal file; forwarded to
        :func:`load_session_files`.
    settings_dict : dict
        Pipeline settings dict. In addition to the keys required by
        :func:`load_session_files`, must contain ``'JSON_PATH'`` (directory
        holding/receiving the sessionData JSON files), ``'ANALYSIS_ID'``,
        ``'TRIAL_OR_RESPONSE_ALIGNED'`` (must already be resolved to a single
        string by the caller, not a list), and
        ``'PIPELINE_SWITCHBOARD'['output_sessionData_json']`` (bool gating
        whether the JSON is written to disk).

    Returns
    -------
    subj_date : str or None
        See :func:`load_session_files`; ``None`` if key-file resolution failed.
    info_key_times : pandas.DataFrame or None
        See :func:`load_session_files`.
    spout_key_times : pandas.DataFrame or None
        See :func:`load_session_files`.
    trial_types : list of str or None
        See :func:`load_session_files`.
    cur_sessionData : dict or None
        The (possibly newly created) sessionData structure, updated in
        memory with this session/alignment/trial-type bookkeeping; ``None``
        if key-file resolution failed.

    Notes
    -----
    Safe to call more than once for the same recording/alignment (e.g. once
    from :func:`~helpers.extract_FP_trial_zscores.run_zscore_extraction` and
    again from
    :func:`~helpers.extract_FP_trial_zscores.run_zscore_extraction_pumpTriggered_spoutOff`):
    trial types that already have measurement data populated by a prior call
    to :func:`~helpers.plotting_and_measurements_helpers.measure_signals_and_save`
    are left untouched rather than reset to ``{}``.
    """
    subj_date, info_key_times, spout_key_times, trial_types, key_file_name = \
        load_session_files(memory_path, settings_dict)

    if subj_date is None:
        return None, None, None, None, None

    json_dir = settings_dict['JSON_PATH']
    analysis_id = settings_dict['ANALYSIS_ID']
    t_or_r_aligned = settings_dict['TRIAL_OR_RESPONSE_ALIGNED']

    json_filenames = glob(json_dir + sep + '*json')
    cur_sessionData_name = json_dir + sep + subj_date + '_sessionData.json'
    if cur_sessionData_name in json_filenames:
        with open(cur_sessionData_name, 'r') as json_file:
            cur_sessionData = json.load(json_file)
    else:
        cur_sessionData = {
            'Subject': subj_date.rsplit('_', 1)[0],
            'Date': subj_date.rsplit('_', 1)[1],
            'AnalysisID': {analysis_id: {'Sessions': [], 'Alignment': {}}}
        }

    if analysis_id not in cur_sessionData['AnalysisID']:
        cur_sessionData['AnalysisID'].update({analysis_id: {'Sessions': [], 'Alignment': {}}})

    session_id = key_file_name[:-4]
    if session_id not in cur_sessionData['AnalysisID'][analysis_id]['Sessions']:
        cur_sessionData['AnalysisID'][analysis_id]['Sessions'].append(session_id)

    if t_or_r_aligned not in cur_sessionData['AnalysisID'][analysis_id]['Alignment']:
        cur_sessionData['AnalysisID'][analysis_id]['Alignment'].update({t_or_r_aligned: {'Trial type': {}}})

    cur_sessionData['AnalysisID'][analysis_id]['Alignment'][t_or_r_aligned]['Settings'] = settings_dict

    for trial_type in trial_types:
        # Only initialize trial types that aren't already present — this function is
        # called again (e.g. by the extinction/pump-triggered path) after another stage
        # may have already populated real measurement data here via measure_signals_and_save;
        # unconditionally resetting to {} would silently wipe that data out.
        if trial_type not in cur_sessionData['AnalysisID'][analysis_id]['Alignment'][t_or_r_aligned]['Trial type']:
            cur_sessionData['AnalysisID'][analysis_id]['Alignment'][t_or_r_aligned]['Trial type'][trial_type] = {}

    if settings_dict['PIPELINE_SWITCHBOARD']['output_sessionData_json']:
        write_json(cur_sessionData, json_dir,
                   cur_sessionData['Subject'] + '_' + cur_sessionData['Date'] + '_sessionData.json')

    return subj_date, info_key_times, spout_key_times, trial_types, cur_sessionData