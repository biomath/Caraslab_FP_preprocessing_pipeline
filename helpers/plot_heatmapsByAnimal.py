"""Builds a cross-subject/cross-session heatmap PDF of trial-summary curves, one panel per trial type."""
from os import makedirs
from os.path import sep
import platform

from glob import glob
from re import split

import numpy as np
from matplotlib import pyplot as plt, patches
from matplotlib import colormaps

import pandas as pd

from matplotlib.backends.backend_pdf import PdfPages

# Tweak the regex file separator for cross-platform compatibility
if platform.system() == 'Windows':
    REGEX_SEP = sep * 2
else:
    REGEX_SEP = sep

def plot_heatmapsByAnimal(SETTINGS_DICT):
    """Build a per-trial-type heatmap PDF of mean signal across all subjects/sessions.

    Loads every ``*_trialSummary_curves.csv`` under
    ``{OUTPUT_PATH}/Aligned signals`` (as produced by
    :func:`helpers.plotting_and_measurements_helpers.plot_FP_trial_zscore_summary`),
    optionally filtered by session name via ``SESSIONS_TO_RUN``/``SESSIONS_TO_EXCLUDE``,
    uniformizes/bins their signal lengths, fills in missing (subject, day,
    trial type) combinations with NaN rows, sorts sessions by mean response
    in the ``SORT_BY_WHICH_TRIALTYPE`` window, and plots one heatmap panel
    per trial type (rows = sessions, columns = time) with a per-subject
    color sidebar, to a single multi-panel PDF page.

    Parameters
    ----------
    SETTINGS_DICT : dict
        Pipeline settings dict. Reads:

        - ``'OUTPUT_PATH'`` : base output directory; input CSVs are read
          from and the heatmap PDF is written to
          ``{OUTPUT_PATH}/Aligned signals`` and ``OUTPUT_PATH`` respectively.
        - ``'EXPERIMENT_TAG'`` : used in the output PDF filename.
        - ``'SUBJECT_COLORS'`` : list of colors for the per-subject sidebar,
          or ``None`` to fall back to the ``'tab20'`` colormap.
        - ``'SESSIONS_TO_RUN'`` : list of session-name substrings to include,
          a path to a CSV with a ``'Session'`` column, or ``None`` for all.
        - ``'SESSIONS_TO_EXCLUDE'`` : list of session-name substrings to
          exclude, or ``None``.
        - ``'BIN_SIZE'`` : time-bin size (seconds) used to convert
          time-window bounds to sample-index bounds.
        - ``'PLOT_PRETRIAL_DURATION'`` / ``'PLOT_POSTTRIAL_DURATION'`` :
          seconds of signal to show before/after the alignment point.
        - ``'TRIALTYPE_DICT'`` : mapping of trial-type name to a
          ``(snippet_start, snippet_end)`` window (seconds) used for sorting.
        - ``'SORT_BY_WHICH_TRIALTYPE'`` : which trial type's window/mean
          response is used to order sessions within each subject group.
        - ``'SAMPLING_RATE'`` : recording sampling rate, or ``None`` to
          estimate it from the CSV's ``'Time_s'`` column.

    Returns
    -------
    None
        Side effect only: writes ``{EXPERIMENT_TAG}_heatmap.pdf`` to
        ``OUTPUT_PATH``.

    Raises
    ------
    FileNotFoundError
        If no ``*trialSummary_curves.csv`` files are found under
        ``{OUTPUT_PATH}/Aligned signals`` after applying the
        ``SESSIONS_TO_RUN``/``SESSIONS_TO_EXCLUDE`` filters.
    """
    # Load globals
    output_path = SETTINGS_DICT['OUTPUT_PATH']
    input_path = output_path + sep + 'Aligned signals'
    experiment_tag = SETTINGS_DICT['EXPERIMENT_TAG']
    subject_colors = SETTINGS_DICT['SUBJECT_COLORS']

    sessions_to_run = SETTINGS_DICT['SESSIONS_TO_RUN']

    sessions_to_exclude = SETTINGS_DICT['SESSIONS_TO_EXCLUDE']
    trial_zscore_plots_path = output_path + sep + 'Aligned signals'
    makedirs(trial_zscore_plots_path, exist_ok=True)
    bin_size = SETTINGS_DICT['BIN_SIZE']
    plot_pretrial_duration = SETTINGS_DICT['PLOT_PRETRIAL_DURATION']
    plot_posttrial_duration = SETTINGS_DICT['PLOT_POSTTRIAL_DURATION']
    trialType_dict = SETTINGS_DICT['TRIALTYPE_DICT']
    sort_by_which_trialtype = SETTINGS_DICT['SORT_BY_WHICH_TRIALTYPE']

    signals_path = glob(input_path + sep + '*trialSummary_curves.csv')

    if sessions_to_run is not None:
        if sep in sessions_to_run:  # A file with session names was specified
            sessions_to_run = pd.read_csv(sessions_to_run)['Session'].values

        signals_path = [path for path in signals_path if
                        any([chosen for chosen in sessions_to_run if chosen in path])]

    if sessions_to_exclude is not None:
        signals_path = [path for path in signals_path if
                        not any([chosen for chosen in sessions_to_exclude if chosen in path])]

    if len(signals_path) == 0:
        raise FileNotFoundError(
            'No *trialSummary_curves.csv files found in ' + input_path +
            ' (after applying SESSIONS_TO_RUN/SESSIONS_TO_EXCLUDE filters, if any). '
            'Check OUTPUT_PATH and the session filters in SETTINGS_DICT.')

    day_list = []
    subj_list = []
    sig_list = []
    # time_list = []
    trialType_list = []
    cur_data = []
    print("Loading data in CSVs...")
    for file_name in signals_path:
        split_file_name = split(REGEX_SEP, file_name)[-1]
        cur_subj = split('_*_', split_file_name)[0]
        cur_data = pd.read_csv(file_name)
        for trial_type in set(cur_data['Trial_type']):
            cur_subj_day = list(set(cur_data[cur_data['Trial_type'] == trial_type]['Recording']))[0]
            trialType_list.append(trial_type)
            day_list.append(cur_subj_day)
            subj_list.append(cur_subj)
            cur_sigs = cur_data[cur_data['Trial_type'] == trial_type]
            # Bound signals by plot brackets
            cur_sigs = cur_sigs[(cur_sigs['Time_s'] >= -plot_pretrial_duration) & (cur_sigs['Time_s'] <= plot_posttrial_duration)]
            sig_list.append(cur_sigs['Signal_mean'].values)

    if SETTINGS_DICT['SAMPLING_RATE'] is None:
        sampling_interval = np.mean(np.diff(cur_data[cur_data['Trial_type'] == trial_type]['Time_s']))
    else:
        sampling_interval = 1 / SETTINGS_DICT['SAMPLING_RATE']

    # Now uniformize lengths (tolerated jitter of 1 point) and resample using bin_size
    print('Uniformizing signal lengths for plotting...')
    min_length = np.round(np.min([len(x) for x in sig_list if not np.all(np.isnan(x))]), 0)
    # plot_list = np.array([resample(x[0:int(min_length)], int(min_length*sampling_interval/bin_size)) for x in sig_list], dtype=float)
    plot_list = np.array(
        [x[0:int(min_length)] for x in sig_list], dtype=float)

    print('Done!')

    # Color cluster separation
    unique_groups = np.sort(np.unique(subj_list))

    # Add missing sessions as NaN
    print('Adding missing sessions as NaN...')
    for group_idx, cur_group in enumerate(unique_groups):
        if sessions_to_run is not None:
            cur_sessions = [s for s in sessions_to_run if cur_group in s]
        else:
            cur_sessions = sorted(list(set([s for s in day_list if cur_group in s])))

        for subj_session in cur_sessions:
            cur_day_filter = np.in1d(day_list, subj_session)
            cur_day_trials = np.unique(np.array(trialType_list)[cur_day_filter])
            for required_trial in trialType_dict.keys():
                if required_trial not in cur_day_trials:
                    day_list.append(subj_session)
                    trialType_list.append(required_trial)
                    subj_list.append(cur_group)
                    to_append = np.empty(np.size(plot_list, axis=1))
                    to_append[:] = np.NaN
                    plot_list = np.vstack((plot_list, to_append))
    print('Done!')

    # Organize and sort signals for plotting
    print('Organizing and sorting signals for plotting...')
    overall_session_order = list()
    colorbar_splits = [0,]
    for group_idx, cur_group in enumerate(unique_groups):
        if cur_group == 'all':
            cur_sessions = day_list
        else:
            if sessions_to_run is not None:
                cur_sessions = [s for s in sessions_to_run if cur_group in s]
            else:
                cur_sessions = sorted(list(set([s for s in day_list if cur_group in s])))

        # cur_units = UNITS_TO_RUN['Unit'].values
        cur_day_filter = np.in1d(day_list, cur_sessions)
        cur_trial_filter = np.array(trialType_list) == sort_by_which_trialtype
        day_trial_filter = cur_day_filter * cur_trial_filter

        cur_sessions = np.array(day_list)[day_trial_filter]
        cur_resps = plot_list[day_trial_filter]

        snippet_start = trialType_dict[sort_by_which_trialtype][0]
        snippet_end = trialType_dict[sort_by_which_trialtype][1]
        # plot_list columns start at time -plot_pretrial_duration, so shifting the
        # window bounds by +plot_pretrial_duration before dividing by bin_size converts
        # them from seconds-relative-to-alignment into column indices into plot_list.
        relevant_indices = np.arange(
            np.floor((snippet_start + plot_pretrial_duration) / bin_size),
            np.floor((snippet_end + plot_pretrial_duration) / bin_size) )

        # NOTE: cur_auroc is a holdover name from a different (auROC-based) pipeline;
        # here it's just one session's heatmap signal row.
        relevant_snippet = np.array([cur_auroc[[int(idx) for idx in relevant_indices]]
                                     for cur_auroc in cur_resps])

        sorted_indices = np.argsort([np.mean(x) for x in relevant_snippet])[::-1]

        overall_session_order.extend(cur_sessions[sorted_indices])
        colorbar_splits.append(len(overall_session_order))

    # Heatmap rows are drawn top-to-bottom in overall_session_order, but the subject
    # colorbar below is built bottom-to-top (axes-fraction y=0 is the bottom), so the
    # cumulative session counts are flipped here to match that drawing order.
    colorbar_splits = [len(overall_session_order) - x for x in colorbar_splits]
    print('Done!')

    print('Plotting...')
    f = plt.figure()
    plot_location_idx = 1
    with PdfPages(sep.join([output_path, experiment_tag + '_heatmap.pdf'])) as pdf:
        for group_idx, cur_group in enumerate(trialType_dict.keys()):
            ax = f.add_subplot(1, len(trialType_dict.keys()), plot_location_idx)

            cur_trial_filter = np.array(trialType_list) == cur_group
            cur_resps = plot_list[cur_trial_filter]

            cur_day_unordered = np.array(day_list)[cur_trial_filter]

            cur_sessions_ordered = []
            for cur_session in overall_session_order:
                cur_sessions_ordered.extend(np.where(cur_day_unordered == cur_session)[0])

            sorted_plot_list = cur_resps[np.array(cur_sessions_ordered)]
            n_sessions = np.size(sorted_plot_list, axis=0)

            current_cmap = colormaps['plasma']
            current_cmap.set_bad('white')
            cax = ax.imshow(sorted_plot_list, vmin=-1, vmax=10, interpolation='None', cmap=current_cmap,
                            extent=[0, np.size(sorted_plot_list, axis=1), n_sessions, 0],
                            aspect='auto')

            ax.axvline(x=plot_pretrial_duration / bin_size, color='white', linestyle='--')

            # Add subject colors
            # These patches start at the bottom left corner so we need to use the reversed lists
            if subject_colors is None:
                subject_colors = colormaps['tab20'].colors

            for group_idx, cur_group in reversed(list(enumerate(unique_groups))):
                y_start = (colorbar_splits[group_idx + 1]) / n_sessions
                y_height = (colorbar_splits[group_idx]) / n_sessions - y_start
                # x=-0.1, width=0.1 in axes-fraction coords (transform=ax.transAxes) draws
                # a sidebar strip 10% of the axes' width just to the left of the heatmap;
                # clip_on=False keeps it visible outside the axes' data area.
                rect = patches.Rectangle(
                    (-0.1, y_start), width=0.1, height=y_height, facecolor=subject_colors[group_idx], transform=ax.transAxes,
                    clip_on=False, edgecolor='none'
                )
                ax.add_patch(rect)

            # if plot_location_idx > 1:
            ax.yaxis.set_visible(False)

            ax.xaxis.set_ticks_position('bottom')

            ax.set_xticks(
                np.arange(0, (plot_pretrial_duration + plot_posttrial_duration + 1) / bin_size, 20))
            ax.set_xlim(
                [0, (plot_pretrial_duration + plot_posttrial_duration) / bin_size])

            new_ticks = np.round(ax.get_xticks() * bin_size - plot_pretrial_duration, 1)
            ax.set_xticklabels(new_ticks.astype(int))
            ax.set_xlabel('Time (s)')

            ax.tick_params(axis='x', bottom=False)
            ax.tick_params(axis='y', left=False, right=False)

            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)

            plot_location_idx += 1
        plt.subplots_adjust(wspace=0.2)
        pdf.savefig()
        plt.close()
    print('Done!')
