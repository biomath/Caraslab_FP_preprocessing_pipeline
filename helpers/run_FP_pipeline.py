"""Thin per-unit-of-work dispatcher that gates z-score extraction based on the pipeline switchboard."""
from platform import system
from os import remove, makedirs
from os.path import sep
from re import split
from copy import deepcopy
from glob import glob
from datetime import datetime
from multiprocessing import Pool

from helpers.extract_FP_trial_zscores import run_zscore_extraction, run_zscore_extraction_pumpTriggered_spoutOff
from helpers.stopwatch import tic, toc
from helpers.write_json import write_json
from helpers.date_utils import find_date_in_tag

# Tweak the regex file separator for cross-platform compatibility
if system() == 'Windows':
    REGEX_SEP = sep * 2
else:
    REGEX_SEP = sep


def run_pipeline(input_list):
    """Run one (subject, date, alignment) unit of the pipeline, as gated by the settings switchboard.

    Intended to be called once per unit of work, either directly or via
    ``multiprocessing.Pool.map`` (hence the single packed-tuple argument).
    Sets ``TRIAL_OR_RESPONSE_ALIGNED`` in a deep-copied settings dict to
    ``run_type`` so downstream functions don't each need to resolve it, then
    calls :func:`~helpers.extract_FP_trial_zscores.run_zscore_extraction`
    if either the ``'extract_trial_zscores'`` or ``'plot_AMDepth_zscores'``
    switchboard flag is set, and
    :func:`~helpers.extract_FP_trial_zscores.run_zscore_extraction_pumpTriggered_spoutOff`
    if either ``'extract_extinction_spoutOff_zscores'`` or
    ``'plot_extinction_spoutOff_zscores'`` is set.

    Parameters
    ----------
    input_list : tuple of (list of str, dict, str)
        Packed positional argument: ``(session_date_paths, SETTINGS_DICT,
        run_type)``.

        session_date_paths : list of str
            Recording paths for this (subject, date) unit.
        SETTINGS_DICT : dict
            Pipeline settings; must include ``'PIPELINE_SWITCHBOARD'`` (dict
            of boolean flags gating which stages run).
        run_type : str
            Alignment mode for this run (e.g. ``'trial'`` or ``'response'``),
            written into the settings dict as ``'TRIAL_OR_RESPONSE_ALIGNED'``.

    Returns
    -------
    None

    Notes
    -----
    ``run_zscore_extraction_pumpTriggered_spoutOff`` itself still contains
    known latent bugs (invalid tuple-style pandas indexing, a ``len()`` call
    on an already-reduced numpy scalar, version-fragile ``scipy.stats.mode``
    usage — see its own docstring) and its downstream
    ``measure_extinction_spoutOffset_signals_and_save`` call passes one
    fewer positional argument than that function requires. Now that this
    dispatch path is reachable, exercising the extinction/pump-triggered
    switchboard flags will hit those bugs; they have not been fixed here.
    """
    # Gather settings
    _, SETTINGS_DICT, run_type = input_list
    settings_dict = deepcopy(SETTINGS_DICT)          # deep copy to avoid mutating the dictionary

    pipeline_switchboard = settings_dict['PIPELINE_SWITCHBOARD']
    settings_dict['TRIAL_OR_RESPONSE_ALIGNED'] = run_type  # Just so I don't need to update it in all functions

    input_list = (input_list[0], settings_dict)

    if pipeline_switchboard['extract_trial_zscores'] or pipeline_switchboard['plot_AMDepth_zscores']:
        run_zscore_extraction(input_list)
    
    # DRAFT CODE: extinction / pump-triggered spout-off pipeline. Not fully
    # validated — see run_zscore_extraction_pumpTriggered_spoutOff's docstring
    # for known bugs. Treat this branch as a work in progress.
    if pipeline_switchboard['extract_extinction_spoutOff_zscores'] or pipeline_switchboard['plot_extinction_spoutOff_zscores']:
        run_zscore_extraction_pumpTriggered_spoutOff(input_list)


def run_full_pipeline(settings_dict):
    """Discover recordings, group them into per-day units, and dispatch :func:`run_pipeline` over all of them.

    This is the notebook-level driver: for every alignment mode in
    ``settings_dict['TRIAL_OR_RESPONSE_ALIGNED']``, it locates the recording
    signal files, sets up the output directories, logs a timestamped snapshot
    of ``settings_dict``, filters sessions per ``SESSIONS_TO_RUN``/
    ``SESSIONS_TO_EXCLUDE``, groups same-day recordings together (if
    ``CONCAT_SAME_DAY``) using :func:`~helpers.date_utils.find_date_in_tag`,
    and finally calls :func:`run_pipeline` for each unit — either directly
    (``DEBUG_RUN``) or in parallel via ``multiprocessing.Pool.map``.

    Parameters
    ----------
    settings_dict : dict
        Pipeline settings dict (typically the notebook's ``SETTINGS_DICT``).
        Mutated in place: ``'OUTPUT_PLOTS_PATH'`` is (re)set on it for each
        alignment mode. Must include ``'TRIAL_OR_RESPONSE_ALIGNED'`` (str or
        list), ``'SIGNALS_PATH'``, ``'OUTPUT_PATH'``, ``'ANALYSIS_ID'``,
        ``'SESSIONS_TO_RUN'``, ``'SESSIONS_TO_EXCLUDE'``, ``'CONCAT_SAME_DAY'``,
        ``'DEBUG_RUN'``, and ``'NUMBER_OF_CORES'``.

    Returns
    -------
    None
        Side effects only: creates output directories, writes a
        ``settings_dict_<timestamp>.json`` snapshot per alignment mode,
        clears stale ``*_tempfile_*.csv`` files, prints progress, and runs
        :func:`run_pipeline` over every discovered (subject, date) unit.

    Raises
    ------
    UserWarning
        If no signal files are found under ``settings_dict['SIGNALS_PATH']``
        (either none exist, or the ``SESSIONS_TO_RUN``/``SESSIONS_TO_EXCLUDE``
        filters excluded all of them), or if either filter is set to a value
        that isn't a ``str`` or ``list``.
    """
    if type(settings_dict['TRIAL_OR_RESPONSE_ALIGNED']) == str:
        trial_or_response_aligned = [settings_dict['TRIAL_OR_RESPONSE_ALIGNED'], ]
    else:  # must be list
        assert type(settings_dict['TRIAL_OR_RESPONSE_ALIGNED']) == list, \
            'TRIAL_OR_RESPONSE_ALIGNED must be a string or a list'
        trial_or_response_aligned = settings_dict['TRIAL_OR_RESPONSE_ALIGNED']

    for run_type in trial_or_response_aligned:
        # Start master timer
        t0 = tic()

        signals_path = glob(settings_dict['SIGNALS_PATH'] + sep + '*dff.csv')

        # Create output path
        settings_dict['OUTPUT_PLOTS_PATH'] = settings_dict['OUTPUT_PATH'] + sep + 'Aligned signals - ' + \
            settings_dict['ANALYSIS_ID'] + sep + run_type
        makedirs(settings_dict['OUTPUT_PLOTS_PATH'], exist_ok=True)

        # Log the settings dictionary for reference
        write_json(settings_dict, settings_dict['OUTPUT_PLOTS_PATH'],
                   'settings_dict_' + datetime.now().strftime("%m%d%y%H%M%S") + '.json')

        # Create directory for session data json files
        makedirs(settings_dict['OUTPUT_PATH'] + sep + 'JSON files', exist_ok=True)

        # Clear older multiprocessing temp files if they exist
        process_tempfiles = glob(settings_dict['OUTPUT_PATH'] + sep + '*_tempfile_*.csv')
        [remove(f) for f in process_tempfiles]

        if settings_dict['SESSIONS_TO_RUN'] is not None:
            if isinstance(settings_dict['SESSIONS_TO_RUN'], str):
                signals_path = [path for path in signals_path if settings_dict['SESSIONS_TO_RUN'] in path]
            elif isinstance(settings_dict['SESSIONS_TO_RUN'], list):
                signals_path = [path for path in signals_path if
                                any([chosen for chosen in settings_dict['SESSIONS_TO_RUN'] if chosen in path])]
            else:
                raise UserWarning('No signal files were found. Please check your inclusion/exclusion lists.')

        if settings_dict['SESSIONS_TO_EXCLUDE'] is not None:
            if isinstance(settings_dict['SESSIONS_TO_EXCLUDE'], str):
                signals_path = [path for path in signals_path if settings_dict['SESSIONS_TO_RUN'] not in path]
            elif isinstance(settings_dict['SESSIONS_TO_EXCLUDE'], list):
                signals_path = [path for path in signals_path if
                                not any([chosen for chosen in settings_dict['SESSIONS_TO_EXCLUDE'] if chosen in path])]
            else:
                raise UserWarning('No signal files were found. Please check your inclusion/exclusion lists.')

        if len(signals_path) == 0:
            raise UserWarning('No signal files were found. Please check your paths.')

        # Identify sessions from the same subject in a day if desired. Otherwise run one session at a time
        # Generate a list of inputs to be passed to each worker
        input_lists = list()
        run_list = list()
        if settings_dict['CONCAT_SAME_DAY']:
            # Identify sessions from the same subject in a day
            subj_session_list = [split('_*_', split(REGEX_SEP, path)[-1])[0:2] for path in signals_path]
            date_list = [find_date_in_tag(subj_session[1]) for subj_session in subj_session_list]
            run_list = [(subj_session[0], cur_date) for subj_session, cur_date in zip(subj_session_list, date_list)]
        else:
            run_list = signals_path

        print(f'>>> START {run_type}: signals={len(run_list) if "signals_path" in dir() else "NA"}')

        cur_date_paths_list = list()
        for dummy_idx, unique_runID in enumerate(run_list):
            if settings_dict['CONCAT_SAME_DAY']:
                (unique_subj, unique_date) = unique_runID
                subj_paths = [path for path in signals_path if unique_subj in path]
                cur_date_paths = [path for path in subj_paths if unique_date in path]
            else:
                cur_date_paths = (unique_runID,)

            # Avoid duplicated runs
            if cur_date_paths in cur_date_paths_list:
                continue
            else:
                cur_date_paths_list.append(cur_date_paths)

        for cur_date_paths in cur_date_paths_list:
            if settings_dict['DEBUG_RUN']:
                run_pipeline((cur_date_paths, settings_dict, run_type))
            else:
                input_lists.append((cur_date_paths, settings_dict, run_type))

        if not settings_dict['DEBUG_RUN']:
            pool = Pool(settings_dict['NUMBER_OF_CORES'])

            # # Feed each worker with all memory paths from one unit
            pool_map_result = pool.map(run_pipeline, input_lists)

            pool.close()

            pool.join()

        toc(t0)
        print(f'<<< END {run_type}: input_lists={len(input_lists)}, '
              f'cur_date_paths_list={len(cur_date_paths_list)}')