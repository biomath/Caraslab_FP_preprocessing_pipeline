"""Plots summary/AM-depth/extinction z-score curves and computes per-trial AUC/peak measurements.

Consumes the per-trial dF/F and z-score arrays assembled by
:mod:`helpers.extract_FP_trial_zscores`, produces PDF plots and CSV curve
exports, measures AUC/peak values per trial and writes them into the
session JSON, and exports a flat CSV formatted for FLMM analysis.
"""
from os.path import sep

import numpy as np
from scipy.integrate import simpson
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from matplotlib import patches, colormaps
import csv

from helpers.format_axes import format_ax
from helpers.write_json import write_json


def plot_FP_trial_zscore_summary(trial_type_dict, align_to_response, subj_date,
                                 output_plots_path, target_sound_onset, target_sound_offset,
                                 paradigm_type, response_latency_filter,
                                 signal_start, signal_end,
                                 ams_to_analyze, _precision_decimals,
                                 all_hit_color, hitShock_color, hitNoShock_color, miss_color, missShock_color,
                                 missNoShock_color, fa_color, reject_color, passivePre_color, passivePost_color):
    """Plot a single-panel summary of mean z-scored signal (± SEM) per trial type and export the underlying curves.

    Produces one PDF (one overlaid line per trial type present in
    ``trial_type_dict``) and a companion ``_curves.csv`` with the plotted
    mean/SEM traces.

    Parameters
    ----------
    trial_type_dict : dict
        Mapping of trial-type name to a dict with ``'info'`` (per-trial
        ``[trial_id, amdepth_dB, onset, offset, resp_latency]`` lists) and
        ``'zscore'`` (per-trial z-scored signal arrays), as produced by
        :mod:`helpers.extract_FP_trial_zscores`.
    align_to_response : bool
        If True, label the x-axis/output filenames as response-aligned;
        otherwise trial-onset-aligned.
    subj_date : str
        ``'{subject}_{date}'`` identifier used in output filenames/CSV rows.
    output_plots_path : str
        Directory to write the PDF and CSV into.
    target_sound_onset, target_sound_offset : float
        Sound onset/offset times (seconds), shaded on the plot when
        ``align_to_response`` is False.
    paradigm_type : {'1IFC', 'AversiveAM'}
        Behavioral paradigm; adds an extra response-window shading band for
        ``'1IFC'``.
    response_latency_filter : float
        Minimum response latency used to shade the allowed response window
        for 1IFC plots.
    signal_start, signal_end : float
        Time bounds (seconds) of the extracted signal window, used to build
        the shared x-axis.
    ams_to_analyze : list or None
        If not None, restricts plotted trials to those whose AM depth (index
        1 of each trial's ``'info'`` entry) is in this list.
    _precision_decimals : int
        Number of decimals to round CSV values to.
    all_hit_color, hitShock_color, hitNoShock_color, miss_color, missShock_color, missNoShock_color, fa_color, reject_color, passivePre_color, passivePost_color : str
        Matplotlib color specs used to color each trial type's line/legend
        entry.

    Returns
    -------
    None
        Side effects only: writes a PDF and a CSV to ``output_plots_path``.
    """
    if align_to_response:
        file_name = subj_date + '_responseAligned_trialSummary'
        x_label = "Time from response (s)"
    else:
        file_name = subj_date + '_trialAligned_trialSummary'
        x_label = "Time from trial onset (s)"
    sig_mean_dict = dict()
    with PdfPages(sep.join([output_plots_path, file_name + '.pdf'])) as pdf:
        # fig, ax = plt.subplots(1, 1)
        fig = plt.figure()
        ax = fig.add_subplot(111)

        # Trial onset or shading
        if align_to_response:
            ax.axvline(0, linestyle='--', color='black')
        else:
            ax.axvspan(target_sound_onset, target_sound_offset, ymin=0.05, ymax=0.075,
                       facecolor='black', alpha=0.25)

            if paradigm_type == '1IFC':
                ax.axvspan(response_latency_filter, signal_end,
                           ymin=0.025, ymax=0.05, facecolor='g', alpha=0.25)

        legend_handles = list()
        for trial_type in trial_type_dict.keys():
            # Map to color
            if trial_type == 'Hit':
                cur_color = all_hit_color
            elif trial_type == 'Hit (all)':
                cur_color = all_hit_color
            elif trial_type == 'Hit (shock)':
                cur_color = hitShock_color
            elif trial_type == 'Hit (no shock)':
                cur_color = hitNoShock_color
            elif trial_type == 'Miss':
                cur_color = miss_color
            elif trial_type == 'Miss (shock)':
                cur_color = missShock_color
            elif trial_type == 'Miss (no shock)':
                cur_color = missNoShock_color
            elif trial_type == 'False alarm':
                cur_color = fa_color
            elif trial_type == 'Reject':
                cur_color = reject_color
            elif trial_type == 'Passive-Pre':
                cur_color = passivePre_color
            elif trial_type == 'Passive-Post':
                cur_color = passivePost_color
            else:
                cur_color = 'black'

            all_trial_info = trial_type_dict[trial_type]['info']
            sigs = []
            for i, trial_info in enumerate(all_trial_info):

                ts = trial_type_dict[trial_type]['zscore'][i]

                if ams_to_analyze is not None:
                    if trial_info[1] in ams_to_analyze:
                        sigs.append(ts)
                    else:
                        continue
                else:
                    sigs.append(ts)

            plot_sigs = np.array(sigs)

            if np.size(sigs) == 0:
                continue

            if '(no shock)' in trial_type:
                linestyle = '--'
            else:
                linestyle = '-'

            signals_mean = np.nanmean(plot_sigs, axis=0)
            signals_sem = np.nanstd(plot_sigs, axis=0, ddof=1) / np.sqrt(
                np.count_nonzero(~np.isnan(plot_sigs), axis=0))
            x_axis = np.linspace(signal_start, signal_end, len(signals_mean))
            ax.plot(x_axis, signals_mean, color=cur_color, linestyle=linestyle)
            ax.fill_between(x_axis, signals_mean - signals_sem, signals_mean + signals_sem,
                            alpha=0.1, color=cur_color, edgecolor='none')

            legend_handles.append(patches.Patch(facecolor=cur_color, edgecolor=None, alpha=0.5,
                                                label=trial_type))

            sig_mean_dict.update({trial_type: (x_axis, signals_mean, signals_sem)})

        format_ax(ax)

        ax.set_xlabel(x_label)
        ax.set_ylabel(r'($\Delta$F/F z-score)')

        # Might want to make this a variable
        # ax.set_ylim([-5, 10])

        labels = [h.get_label() for h in legend_handles]

        fig.legend(handles=legend_handles, labels=labels, frameon=False, numpoints=1, bbox_to_anchor=[0.95, 0.95])

        fig.tight_layout()

        # plt.show()
        pdf.savefig()
        plt.close()

    with open(sep.join([output_plots_path, file_name + '_curves.csv']), 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')

        writer.writerow(['Recording'] + ['Trial_type'] + ['Time_s'] + ['Signal_mean'] + [
            'Signal_SEM'])

        for trial_type in sig_mean_dict.keys():
            for time_point_idx, _ in enumerate(sig_mean_dict[trial_type][0]):
                writer.writerow([subj_date] +
                                [trial_type] +
                                [np.round(sig_mean_dict[trial_type][0][time_point_idx], _precision_decimals)] +
                                [np.round(sig_mean_dict[trial_type][1][time_point_idx], _precision_decimals)] +
                                [np.round(sig_mean_dict[trial_type][2][time_point_idx], _precision_decimals)])


def plot_FP_trial_zscore_byAMdepth(trial_type_dict, align_to_response, subj_date,
                                   output_plots_path, target_sound_onset, target_sound_offset,
                                   paradigm_type, response_latency_filter, signal_start, signal_end,
                                   ams_to_analyze, _precision_decimals):
    """Plot mean z-scored signal (± SEM) split by AM depth, one PDF page per trial-type group.

    Groups trial types per ``paradigm_type`` (e.g. combining
    ``'Hit (shock)'``/``'Hit (no shock)'`` onto one page for AversiveAM),
    and within each page draws one colored line per AM depth (color mapped
    via the ``'plasma'`` colormap, from a `perc_value` derived from the dB
    depth). Also exports the underlying curves to a companion CSV.

    Parameters
    ----------
    trial_type_dict : dict
        See :func:`plot_FP_trial_zscore_summary`.
    align_to_response : bool
        See :func:`plot_FP_trial_zscore_summary`.
    subj_date : str
        See :func:`plot_FP_trial_zscore_summary`.
    output_plots_path : str
        See :func:`plot_FP_trial_zscore_summary`.
    target_sound_onset, target_sound_offset : float
        See :func:`plot_FP_trial_zscore_summary`.
    paradigm_type : {'1IFC', 'AversiveAM'}
        Determines which trial-type groups are plotted as separate PDF pages.
    response_latency_filter : float
        See :func:`plot_FP_trial_zscore_summary`.
    signal_start, signal_end : float
        See :func:`plot_FP_trial_zscore_summary`.
    ams_to_analyze : list or None
        If None, AM depths are auto-collected from every trial type present
        in ``trial_type_dict``; otherwise restricts to the given depths.
    _precision_decimals : int
        See :func:`plot_FP_trial_zscore_summary`.

    Returns
    -------
    None
        Side effects only: writes a multi-page PDF and a CSV to
        ``output_plots_path``.
    """
    if align_to_response:
        file_name = subj_date + '_responseAligned_byAMdepth'
        x_label = "Time from response (s)"
    else:
        file_name = subj_date + '_trialAligned_byAMdepth'
        x_label = "Time from trial onset (s)"

    sig_mean_dict = dict()
    # Gather all AMs presented for use in the plotting
    if ams_to_analyze is None:
        all_ams = list()
        for trial_type_key in trial_type_dict.keys():
            all_ams.extend(list(set([x[1] for x in trial_type_dict[trial_type_key]['info']])))
    else:
        all_ams = ams_to_analyze

    with PdfPages(sep.join([output_plots_path, file_name + '.pdf'])) as pdf:
        # Trial grouping for plotting, if you'd like to combine responses
        # Example: trial_groups = [('Hit', 'Reject'), ('Miss', 'False alarm')]
        if paradigm_type == 'AversiveAM':
            # trial_groups = trial_type_dict.keys()
            trial_groups = ['Hit (all)', ('Hit (shock)', 'Hit (no shock)'), ('Miss (shock)', 'Miss (no shock)'),
                            'False alarm', 'Passive-Pre', 'Passive-Post', 'Passive']
        elif paradigm_type == '1IFC':
            trial_groups = ['Hit', 'Reject', 'Miss', 'False alarm', 'Passive']
        else:
            print('Experiment type not recognized. Skipping plotting', flush=True)
            return

        for tgroup in trial_groups:
            fig = plt.figure()
            ax = fig.add_subplot(111)

            # Trial onset or shading
            if align_to_response:
                ax.axvline(0, linestyle='--', color='black')
            else:
                ax.axvspan(target_sound_onset, target_sound_offset, ymin=0.05, ymax=0.075,
                           facecolor='black', alpha=0.25)

                if paradigm_type == '1IFC':
                    ax.axvspan(response_latency_filter, signal_end,
                               ymin=0.025, ymax=0.05, facecolor='g', alpha=0.25)

            cur_trialTypes = list(ttype for ttype in trial_type_dict.keys() if ttype in tgroup)
            legend_handles = list()
            for trial_type in cur_trialTypes:
                if trial_type not in trial_type_dict.keys():
                    continue

                am_sig_list = dict()

                if 'no shock' in trial_type:
                    linestyle = '--'
                else:
                    linestyle = '-'

                for amdepth in sorted(list(set(all_ams)), reverse=True):
                    # amdepth == -40 is the 0%-AM-depth sentinel (see extract_FP_trial_zscores);
                    # perc_value is pinned to 1 here rather than computed via 10**(amdepth/20)
                    # (which would instead round to ~0)
                    if amdepth > -40:
                        perc_value = np.round(10 ** (amdepth / 20), 2)
                    else:
                        perc_value = 1

                    cmap_factor = perc_value

                    cur_color = colormaps.get_cmap('plasma')(int(cmap_factor * 255))

                    sigs = []

                    all_trial_info = trial_type_dict[trial_type]['info']
                    for i, trial_info in enumerate(all_trial_info):
                        ts = trial_type_dict[trial_type]['zscore'][i]
                        if trial_info[1] == amdepth:
                            sigs.append(ts)
                        else:
                            continue

                    plot_sigs = np.array(sigs)

                    if np.size(plot_sigs) == 0 or np.sum(plot_sigs) == 0:
                        continue

                    signals_mean = np.nanmean(plot_sigs, axis=0)
                    signals_sem = np.nanstd(plot_sigs, axis=0, ddof=1) / np.sqrt(
                        np.count_nonzero(~np.isnan(plot_sigs), axis=0))
                    x_axis = np.linspace(signal_start, signal_end,
                                         len(signals_mean))
                    ax.plot(x_axis, signals_mean, color=cur_color, linestyle=linestyle, alpha=1)
                    ax.fill_between(x_axis, signals_mean - signals_sem, signals_mean + signals_sem,
                                    alpha=0.1, color=cur_color, edgecolor='none')

                    legend_handles.append(patches.Patch(facecolor=cur_color, edgecolor=None, alpha=1,
                                                        label=str(amdepth) + ' dB'))

                    am_sig_list.update({amdepth: (x_axis, signals_mean, signals_sem)})
                sig_mean_dict.update({trial_type: am_sig_list})

            format_ax(ax)

            ax.set_xlabel(x_label)
            ax.set_ylabel(r'$\Delta$F/F z-score')

            # Might want to make this a variable
            # ax.set_ylim([-1, 1])

            labels = [h.get_label() for h in legend_handles]

            fig.legend(handles=legend_handles, labels=labels, frameon=False, numpoints=1,
                       bbox_to_anchor=[0.95, 0.95])

            # Plot title
            if all(['Hit' in temp_ttype for temp_ttype in tgroup]):
                fig.suptitle('Hits: shock on (-) vs off (--)')
            elif all(['Miss' in temp_ttype for temp_ttype in tgroup]):
                fig.suptitle('Misses: shock on (-) vs off (--)')
            else:
                if isinstance(tgroup, str):
                    fig.suptitle(tgroup)
                else:
                    fig.suptitle(tgroup[0])
            fig.tight_layout()

            # plt.show()
            pdf.savefig()
            plt.close()

    with open(sep.join([output_plots_path, file_name + '_curves.csv']), 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')

        writer.writerow(['Recording'] + ['Trial_type'] + ['AMDepth'] + ['Time_s'] + ['Signal_mean'] + [
            'Signal_SEM'])

        for trial_type in sig_mean_dict.keys():
            for amdepth in sig_mean_dict[trial_type]:
                for time_point_idx, _ in enumerate(sig_mean_dict[trial_type][amdepth][0]):
                    writer.writerow([subj_date] +
                                    [trial_type] +
                                    [amdepth] +
                                    [np.round(sig_mean_dict[trial_type][amdepth][0][time_point_idx],
                                              _precision_decimals)] +
                                    [np.round(sig_mean_dict[trial_type][amdepth][1][time_point_idx],
                                              _precision_decimals)] +
                                    [np.round(sig_mean_dict[trial_type][amdepth][2][time_point_idx],
                                              _precision_decimals)])



# ──────────────────────────────────────────────────────────────────────────────
# DRAFT CODE: extinction / pump-triggered spout-off pipeline.
# Only exercised via the not-fully-validated
# run_zscore_extraction_pumpTriggered_spoutOff path — treat as a work in progress.
# ──────────────────────────────────────────────────────────────────────────────
def plot_FP_extinction_spoutOff_zscores(trial_type_dict, subj_date,
                                            output_plots_path, signal_start,
                                            signal_end, _precision_decimals):
    """Plot mean z-scored signal (± SEM) aligned to pump zero-flow-triggered spout offset, for the ``'Zero_flow'`` trial type.

    Parameters
    ----------
    trial_type_dict : dict
        Must contain a ``'Zero_flow'`` key with ``'info'`` and ``'zscore'``
        entries, as produced by
        :func:`helpers.extract_FP_trial_zscores.run_zscore_extraction_pumpTriggered_spoutOff`.
    subj_date : str
        See :func:`plot_FP_trial_zscore_summary`.
    output_plots_path : str
        See :func:`plot_FP_trial_zscore_summary`.
    signal_start, signal_end : float
        See :func:`plot_FP_trial_zscore_summary`.
    _precision_decimals : int
        See :func:`plot_FP_trial_zscore_summary`.

    Returns
    -------
    None
        Side effects only: writes a single-page PDF and a CSV to
        ``output_plots_path``.
    """
    trial_type = 'Zero_flow'
    x_label = "Time from response (s)"
    file_name = subj_date + '_extinction_spoutOffAligned'
    with PdfPages(sep.join([output_plots_path, file_name + '.pdf'])) as pdf:
        fig = plt.figure()
        ax = fig.add_subplot(111)

        # Spout offset
        ax.axvline(0, linestyle='--', color='black')

        am_sig_list = dict()
        linestyle = '-'
        cur_color = 'black'

        sigs = []
        all_trial_info = trial_type_dict[trial_type]['info']
        for i, trial_info in enumerate(all_trial_info):
            ts = trial_type_dict[trial_type]['zscore'][i]
            sigs.append(ts)

        plot_sigs = np.array(sigs)

        signals_mean = np.nanmean(plot_sigs, axis=0)
        signals_sem = np.nanstd(plot_sigs, axis=0, ddof=1) / np.sqrt(
            np.count_nonzero(~np.isnan(plot_sigs), axis=0))
        x_axis = np.linspace(signal_start, signal_end,
                             len(signals_mean))
        ax.plot(x_axis, signals_mean, color=cur_color, linestyle=linestyle, alpha=1)
        ax.fill_between(x_axis, signals_mean - signals_sem, signals_mean + signals_sem,
                        alpha=0.1, color=cur_color, edgecolor='none')

        format_ax(ax)

        ax.set_xlabel(x_label)
        ax.set_ylabel(r'$\Delta$F/F z-score')

        fig.tight_layout()

        # plt.show()
        pdf.savefig()
        plt.close()

    with open(sep.join([output_plots_path, file_name + '_curves.csv']), 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')

        writer.writerow(['Recording'] + ['Trial_type'] + ['Time_s'] + ['Signal_mean'] + [
            'Signal_SEM'])

        for time_point_idx, _ in enumerate(signals_mean):
            writer.writerow([subj_date] +
                            [trial_type] +
                            [np.round(x_axis[time_point_idx],
                                      _precision_decimals)] +
                            [np.round(signals_mean[time_point_idx],
                                      _precision_decimals)] +
                            [np.round(signals_sem[time_point_idx],
                                      _precision_decimals)])


def __calculate_PeakValue_and_AUC(sigs, trial_info,
                                  baseline_start, baseline_end,
                                  auc_start, auc_end,
                                  x_axis):
    """Compute per-trial response AUC, peak value, and baseline AUC via Simpson's rule integration.

    Parameters
    ----------
    sigs : numpy.ndarray
        2-D array of shape ``(n_trials, n_timepoints)`` with one signal
        (dF/F or z-score) row per trial.
    trial_info : list
        Per-trial info list; only its length (``n_trials``) is used here.
    baseline_start, baseline_end : float
        Time bounds (seconds, relative to ``x_axis``) defining the baseline
        integration window.
    auc_start, auc_end : float
        Time bounds (seconds, relative to ``x_axis``) defining the response
        integration window.
    x_axis : numpy.ndarray
        1-D time axis (seconds) matching ``sigs``'s second dimension.

    Returns
    -------
    auc_response : numpy.ndarray
        Per-trial area under the curve within the response window
        (Simpson's rule).
    peak : numpy.ndarray
        Per-trial peak value within the response window — whichever of the
        local max or min has the larger absolute value (so peaks can be
        reported as negative deflections).
    auc_baseline : numpy.ndarray
        Per-trial area under the curve within the baseline window.
    """
    auc_response = np.zeros(np.shape(sigs)[0])
    peak = np.zeros(np.shape(sigs)[0])
    auc_baseline = np.zeros(np.shape(sigs)[0])
    for trial_idx, _ in enumerate(trial_info):
        bounded_response_idx_mask = (x_axis >= auc_start) & (x_axis < auc_end)
        bounded_response_xaxis = x_axis[bounded_response_idx_mask]
        bounded_response = sigs[trial_idx, bounded_response_idx_mask]

        baseline_idx_mask = (x_axis >= baseline_start) & (x_axis < baseline_end)
        bounded_baseline_xaxis = x_axis[baseline_idx_mask]
        bounded_baseline = sigs[trial_idx, baseline_idx_mask]

        auc_response[trial_idx] = simpson(bounded_response, x=bounded_response_xaxis)

        # Peak can be positive or negative
        max_peak = np.max(bounded_response)
        min_peak = np.min(bounded_response)
        if np.abs(max_peak) > np.abs(min_peak):
            peak[trial_idx] = max_peak
        else:
            peak[trial_idx] = min_peak

        auc_baseline[trial_idx] = simpson(bounded_baseline, x=bounded_baseline_xaxis)

    return auc_response, peak, auc_baseline


def measure_signals_and_save(trial_type_dict, cur_sessionData, analysis_id, t_or_r_align, subj_date,
                             output_plots_path,
                             baseline_start, baseline_end,
                             signal_start, signal_end,
                             auc_start, auc_end,
                             min_length, ams_to_analyze, _precision_decimals,
                             output_path, output_sessionData_json):
    """Measure per-trial AUC/peak values, write them (+ raw traces) into the session JSON, and export a trial-by-trial CSV.

    For each trial type, stacks per-trial dF/F and z-score signals into
    fixed-width arrays (padded/truncated to ``min_length``), computes AUC and
    peak measurements via :func:`__calculate_PeakValue_and_AUC`, optionally
    writes the full measurement + raw-trace set into
    ``cur_sessionData['AnalysisID'][analysis_id]['Alignment'][t_or_r_align]['Trial type'][trial_type]``
    and persists it via :func:`~helpers.write_json.write_json`, and always
    writes a flat trial-by-trial measurements CSV.

    Parameters
    ----------
    trial_type_dict : dict
        See :func:`plot_FP_trial_zscore_summary`; must also contain
        ``'dff_signal'`` per trial type.
    cur_sessionData : dict
        Session JSON structure (as produced by
        :func:`helpers.preprocess_files.preprocess_files`) to update in place.
    analysis_id : str
        Key identifying this analysis run within ``cur_sessionData['AnalysisID']``.
    t_or_r_align : {'trial_aligned', 'response_aligned'}
        Alignment mode; also used to choose the output CSV filename suffix.
    subj_date : str
        See :func:`plot_FP_trial_zscore_summary`.
    output_plots_path : str
        Directory to write the trial-by-trial measurements CSV into.
    baseline_start, baseline_end, signal_start, signal_end, auc_start, auc_end : float
        Time-window bounds forwarded to :func:`__calculate_PeakValue_and_AUC`
        and used to build the shared x-axis.
    min_length : int
        Common trial length (in samples) that all signals are padded/
        truncated to before measurement.
    ams_to_analyze : list or None
        If not None, restricts measured trials to the given AM depths (see
        Notes for a caveat).
    _precision_decimals : int
        Number of decimals to round measurements/traces to before writing.
    output_path : str
        Base output directory; the sessionData JSON is written to
        ``output_path/JSON files/``.
    output_sessionData_json : bool
        If True, write the updated ``cur_sessionData`` back to disk for each
        trial type processed.

    Returns
    -------
    None
        Side effects only: mutates ``cur_sessionData`` in place, optionally
        writes it to disk, and always writes a CSV to ``output_plots_path``.

    Notes
    -----
    When ``ams_to_analyze`` filters out a trial, its row in the
    pre-allocated ``dff_sigs``/``zscore_sigs`` arrays (already zero-filled
    before the filter check) is left as all zeros rather than removed, so a
    "phantom" zero-signal trial is still measured and included in the
    output alongside genuine trials.
    """
    if t_or_r_align == 'response_aligned':
        file_name = subj_date + '_responseAligned_trialByTrial'
    else:
        file_name = subj_date + '_trialAligned_trialByTrial'

    output_dict = dict()
    for trial_type in trial_type_dict.keys():
        dff_sigs = np.zeros((len(trial_type_dict[trial_type]['dff_signal']), int(min_length)))
        zscore_sigs = np.zeros((len(trial_type_dict[trial_type]['zscore']), int(min_length)))
        all_trial_info = trial_type_dict[trial_type]['info']
        for i, trial_info in enumerate(all_trial_info):
            ts_dff = trial_type_dict[trial_type]['dff_signal'][i]
            ts_zscore = trial_type_dict[trial_type]['zscore'][i]
            if ams_to_analyze is not None:
                if trial_info[1] in ams_to_analyze:
                    dff_sigs[i, 0:len(ts_dff)] = ts_dff
                    zscore_sigs[i, 0:len(ts_zscore)] = ts_zscore
                else:
                    # NOTE: row i of dff_sigs/zscore_sigs was already zero-filled above
                    # and is not removed here, so this trial is skipped but its all-zero
                    # row still gets measured/exported below (see docstring Notes).
                    continue
            else:
                dff_sigs[i, 0:len(ts_dff)] = ts_dff
                zscore_sigs[i, 0:len(ts_zscore)] = ts_zscore

        if np.size(dff_sigs) == 0 or np.size(zscore_sigs) == 0:
            continue

        # Measure and add measurements to list
        trial_info = trial_type_dict[trial_type]['info']
        x_axis = np.linspace(signal_start, signal_end,
                             np.shape(zscore_sigs)[1])  # just in case

        # Get the dff signal measurements
        auc_response_dff, peak_dff, auc_baseline_dff = __calculate_PeakValue_and_AUC(
            dff_sigs, trial_info, baseline_start=baseline_start, baseline_end=baseline_end,
            auc_start=auc_start, auc_end=auc_end, x_axis=x_axis)

        # Get the z-scored signal measurements
        auc_response_zscore, peak_zscore, auc_baseline_zscore = __calculate_PeakValue_and_AUC(
            zscore_sigs, trial_info, baseline_start=baseline_start, baseline_end=baseline_end,
            auc_start=auc_start, auc_end=auc_end, x_axis=x_axis)

        output_dict.update({trial_type: (trial_info,
                                         auc_response_dff,
                                         peak_dff,
                                         auc_baseline_dff,
                                         auc_response_zscore,
                                         peak_zscore,
                                         auc_baseline_zscore)})

        # Output individual session info, curves and measurements in json files in here for optimization purposes
        if output_sessionData_json:
            # Trial info
            cur_sessionData['AnalysisID'][analysis_id]['Alignment'][t_or_r_align]['Trial type'][trial_type][
                'TrialID'] = \
                [x[0] for x in trial_info]
            cur_sessionData['AnalysisID'][analysis_id]['Alignment'][t_or_r_align]['Trial type'][trial_type][
                'AMdepth'] = \
                [np.round(x[1], 2) for x in trial_info]
            cur_sessionData['AnalysisID'][analysis_id]['Alignment'][t_or_r_align]['Trial type'][trial_type][
                'Trial_onset'] = \
                [np.round(x[2], _precision_decimals) for x in trial_info]
            cur_sessionData['AnalysisID'][analysis_id]['Alignment'][t_or_r_align]['Trial type'][trial_type][
                'Trial_offset'] = \
                [np.round(x[3], _precision_decimals) for x in trial_info]
            cur_sessionData['AnalysisID'][analysis_id]['Alignment'][t_or_r_align]['Trial type'][trial_type][
                'RespLatency'] = \
                [np.round(x[4], _precision_decimals) for x in trial_info]

            # Signal measurements
            cur_sessionData['AnalysisID'][analysis_id]['Alignment'][t_or_r_align]['Trial type'][trial_type][
                'Response_auc_dff'] = (
                np.round(auc_response_dff, _precision_decimals))
            cur_sessionData['AnalysisID'][analysis_id]['Alignment'][t_or_r_align]['Trial type'][trial_type][
                'Response_peak_dff'] = (
                np.round(peak_dff, _precision_decimals))
            cur_sessionData['AnalysisID'][analysis_id]['Alignment'][t_or_r_align]['Trial type'][trial_type][
                'Baseline_auc_dff'] = (
                np.round(auc_baseline_dff, _precision_decimals))
            cur_sessionData['AnalysisID'][analysis_id]['Alignment'][t_or_r_align]['Trial type'][trial_type][
                'Response_auc_zscore'] = (
                np.round(auc_response_zscore, _precision_decimals))
            cur_sessionData['AnalysisID'][analysis_id]['Alignment'][t_or_r_align]['Trial type'][trial_type][
                'Response_peak_zscore'] = (
                np.round(peak_zscore, _precision_decimals))
            cur_sessionData['AnalysisID'][analysis_id]['Alignment'][t_or_r_align]['Trial type'][trial_type][
                'Baseline_auc_zscore'] = (
                np.round(auc_baseline_zscore, _precision_decimals))

            # Transients and time axis
            cur_sessionData['AnalysisID'][analysis_id]['Alignment'][t_or_r_align]['Trial type'][trial_type][
                'Time_s'] = (
                np.round(x_axis, _precision_decimals))
            cur_sessionData['AnalysisID'][analysis_id]['Alignment'][t_or_r_align]['Trial type'][trial_type][
                'Calcium_dff'] = (
                np.round(dff_sigs, _precision_decimals))
            cur_sessionData['AnalysisID'][analysis_id]['Alignment'][t_or_r_align]['Trial type'][trial_type][
                'Calcium_zscore'] = (
                np.round(zscore_sigs, _precision_decimals))

            write_json(cur_sessionData, output_path + sep + 'JSON files',
                       cur_sessionData['Subject'] + '_' + cur_sessionData['Date'] + '_sessionData.json')

    # Write csv with area under curves
    with open(sep.join([output_plots_path, file_name + '.csv']), 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')

        writer.writerow(['Recording'] +
                        ['Trial_type'] +
                        ['TrialID'] +
                        ['AMDepth'] +
                        ['Trial_Onset'] +
                        ['Trial_Offset'] +
                        ['RespLatency'] +
                        ['Area_under_curve_dff'] +
                        ['Peak_value_dff'] +
                        ['Baseline_area_under_curve_dff'] +
                        ['Area_under_curve_zscore'] +
                        ['Peak_value_zscore'] +
                        ['Baseline_area_under_curve_zscore'])

        for trial_type in output_dict.keys():
            for trial_idx in range(len(output_dict[trial_type][0])):
                # output_list[x][0] is (cur_trial['trialid'], cur_trial['amdepth'], cur_trial['trial_onset'])

                trialID = output_dict[trial_type][0][trial_idx][0]
                AMdepth = np.round(output_dict[trial_type][0][trial_idx][1], 2)
                trial_onset = np.round(output_dict[trial_type][0][trial_idx][2], _precision_decimals)
                trial_offset = np.round(output_dict[trial_type][0][trial_idx][3], _precision_decimals)
                resp_latency = np.round(output_dict[trial_type][0][trial_idx][4], _precision_decimals)

                response_auc_dff = np.round(output_dict[trial_type][1][trial_idx], _precision_decimals)
                response_peak_dff = np.round(output_dict[trial_type][2][trial_idx], _precision_decimals)
                baseline_auc_dff = np.round(output_dict[trial_type][3][trial_idx], _precision_decimals)
                response_auc_zscore = np.round(output_dict[trial_type][4][trial_idx], _precision_decimals)
                response_peak_zscore = np.round(output_dict[trial_type][5][trial_idx], _precision_decimals)
                baseline_auc_zscore = np.round(output_dict[trial_type][6][trial_idx], _precision_decimals)

                writer.writerow([subj_date] +
                                [trial_type] +
                                [trialID] +
                                [AMdepth] +
                                [trial_onset] +
                                [trial_offset] +
                                [resp_latency] +
                                [response_auc_dff] +
                                [response_peak_dff] +
                                [baseline_auc_dff] +
                                [response_auc_zscore] +
                                [response_peak_zscore] +
                                [baseline_auc_zscore]
                                )

# ──────────────────────────────────────────────────────────────────────────────
# DRAFT CODE: extinction / pump-triggered spout-off pipeline.
# Known call-site arity bug (see docstring below) — treat as a work in progress.
# ──────────────────────────────────────────────────────────────────────────────
def measure_extinction_spoutOffset_signals_and_save(
        trial_type_dict, cur_sessionData, analysis_id, subj_date,
        output_plots_path,
        baseline_start, baseline_end,
        signal_start, signal_end,
        auc_start, auc_end,
        min_length, _precision_decimals,
        sampling_frequency, output_path, output_sessionData_json):
    """Measure per-trial AUC/peak values for pump-triggered spout-offset ("extinction") trials and export them.

    Analogous to :func:`measure_signals_and_save`, but specialized for the
    single ``'Zero_flow'`` trial type produced by
    :func:`helpers.extract_FP_trial_zscores.run_zscore_extraction_pumpTriggered_spoutOff`
    (no ``ams_to_analyze`` filtering, and always writes under the hardcoded
    ``'response_aligned'`` alignment key).

    Parameters
    ----------
    trial_type_dict : dict
        See :func:`measure_signals_and_save`.
    cur_sessionData : dict
        See :func:`measure_signals_and_save`.
    analysis_id : str
        See :func:`measure_signals_and_save`.
    subj_date : str
        See :func:`plot_FP_trial_zscore_summary`.
    output_plots_path : str
        See :func:`measure_signals_and_save`.
    baseline_start, baseline_end, signal_start, signal_end, auc_start, auc_end : float
        See :func:`measure_signals_and_save`.
    min_length : int
        See :func:`measure_signals_and_save`.
    _precision_decimals : int
        See :func:`measure_signals_and_save`.
    sampling_frequency : float
        Not used in this function's body.
    output_path : str
        See :func:`measure_signals_and_save`.
    output_sessionData_json : bool
        See :func:`measure_signals_and_save`.

    Returns
    -------
    None
        Side effects only, mirroring :func:`measure_signals_and_save`.

    Notes
    -----
    This function's body is ~95% duplicated from :func:`measure_signals_and_save`
    (minus the ``ams_to_analyze`` filter and AM-depth field). Its only
    current call site (inside the dead
    :func:`helpers.extract_FP_trial_zscores.run_zscore_extraction_pumpTriggered_spoutOff`
    code path) passes 15 positional arguments while this signature requires
    16 — if that path were ever reactivated as-is, ``output_path`` would
    land in the ``sampling_frequency`` slot and ``output_sessionData_json``
    would be missing entirely, raising a ``TypeError``.
    """
    t_or_r_align = 'response_aligned'  # Hardcode this for now
    file_name = subj_date + '_extinction_spoutOffAligned_trialByTrial'
    output_dict = dict()
    for trial_type in trial_type_dict.keys():
        dff_sigs = np.zeros((len(trial_type_dict[trial_type]['dff_signal']), int(min_length)))
        zscore_sigs = np.zeros((len(trial_type_dict[trial_type]['zscore']), int(min_length)))
        all_trial_info = trial_type_dict[trial_type]['info']
        for i, trial_info in enumerate(all_trial_info):
            ts_dff = trial_type_dict[trial_type]['dff_signal'][i]
            ts_zscore = trial_type_dict[trial_type]['zscore'][i]

            dff_sigs[i, 0:len(ts_dff)] = ts_dff
            zscore_sigs[i, 0:len(ts_zscore)] = ts_zscore

        if np.size(dff_sigs) == 0 or np.size(zscore_sigs) == 0:
            continue

        # Measure and add measurements to list
        trial_info = trial_type_dict[trial_type]['info']
        x_axis = np.linspace(signal_start, signal_end,
                             np.shape(zscore_sigs)[1])  # just in case

        # Get the dff signal measurements
        auc_response_dff, peak_dff, auc_baseline_dff = __calculate_PeakValue_and_AUC(
            dff_sigs, trial_info, baseline_start=baseline_start, baseline_end=baseline_end,
            auc_start=auc_start, auc_end=auc_end, x_axis=x_axis)

        # Get the z-scored signal measurements
        auc_response_zscore, peak_zscore, auc_baseline_zscore = __calculate_PeakValue_and_AUC(
            zscore_sigs, trial_info, baseline_start=baseline_start, baseline_end=baseline_end,
            auc_start=auc_start, auc_end=auc_end, x_axis=x_axis)

        output_dict.update({trial_type: (trial_info,
                                         auc_response_dff,
                                         peak_dff,
                                         auc_baseline_dff,
                                         auc_response_zscore,
                                         peak_zscore,
                                         auc_baseline_zscore)})

        # Output individual session info, curves and measurements in json files in here for optimization purposes
        if output_sessionData_json:
            # Trial info
            cur_sessionData['AnalysisID'][analysis_id]['Alignment'][t_or_r_align]['Trial type'][trial_type][
                'TrialID'] = \
                [x[0] for x in trial_info]
            cur_sessionData['AnalysisID'][analysis_id]['Alignment'][t_or_r_align]['Trial type'][trial_type][
                'Trial_onset'] = \
                [np.round(x[1], _precision_decimals) for x in trial_info]
            cur_sessionData['AnalysisID'][analysis_id]['Alignment'][t_or_r_align]['Trial type'][trial_type][
                'Trial_offset'] = \
                [np.round(x[2], _precision_decimals) for x in trial_info]
            cur_sessionData['AnalysisID'][analysis_id]['Alignment'][t_or_r_align]['Trial type'][trial_type][
                'RespLatency'] = \
                [np.round(x[3], _precision_decimals) for x in trial_info]

            # Signal measurements
            cur_sessionData['AnalysisID'][analysis_id]['Alignment'][t_or_r_align]['Trial type'][trial_type][
                'Response_auc_dff'] = (
                np.round(auc_response_dff, _precision_decimals))
            cur_sessionData['AnalysisID'][analysis_id]['Alignment'][t_or_r_align]['Trial type'][trial_type][
                'Response_peak_dff'] = (
                np.round(peak_dff, _precision_decimals))
            cur_sessionData['AnalysisID'][analysis_id]['Alignment'][t_or_r_align]['Trial type'][trial_type][
                'Baseline_auc_dff'] = (
                np.round(auc_baseline_dff, _precision_decimals))
            cur_sessionData['AnalysisID'][analysis_id]['Alignment'][t_or_r_align]['Trial type'][trial_type][
                'Response_auc_zscore'] = (
                np.round(auc_response_zscore, _precision_decimals))
            cur_sessionData['AnalysisID'][analysis_id]['Alignment'][t_or_r_align]['Trial type'][trial_type][
                'Response_peak_zscore'] = (
                np.round(peak_zscore, _precision_decimals))
            cur_sessionData['AnalysisID'][analysis_id]['Alignment'][t_or_r_align]['Trial type'][trial_type][
                'Baseline_auc_zscore'] = (
                np.round(auc_baseline_zscore, _precision_decimals))

            # Transients and time axis
            cur_sessionData['AnalysisID'][analysis_id]['Alignment'][t_or_r_align]['Trial type'][trial_type][
                'Time_s'] = (
                np.round(x_axis, _precision_decimals))
            cur_sessionData['AnalysisID'][analysis_id]['Alignment'][t_or_r_align]['Trial type'][trial_type][
                'Calcium_dff'] = (
                np.round(dff_sigs, _precision_decimals))
            cur_sessionData['AnalysisID'][analysis_id]['Alignment'][t_or_r_align]['Trial type'][trial_type][
                'Calcium_zscore'] = (
                np.round(zscore_sigs, _precision_decimals))

            write_json(cur_sessionData, output_path + sep + 'JSON files',
                       cur_sessionData['Subject'] + '_' + cur_sessionData['Date'] + '_sessionData.json')

    # Write csv with area under curves
    with open(sep.join([output_plots_path, file_name + '.csv']), 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')

        writer.writerow(['Recording'] +
                        ['Trial_type'] +
                        ['TrialID'] +
                        ['Trial_Onset'] +
                        ['Trial_Offset'] +
                        ['RespLatency'] +
                        ['Area_under_curve_dff'] +
                        ['Peak_value_dff'] +
                        ['Baseline_area_under_curve_dff'] +
                        ['Area_under_curve_zscore'] +
                        ['Peak_value_zscore'] +
                        ['Baseline_area_under_curve_zscore'])

        for trial_type in output_dict.keys():
            for trial_idx in range(len(output_dict[trial_type][0])):
                # output_list[x][0] is (cur_trial['trialid'], cur_trial['amdepth'], cur_trial['trial_onset'])

                trialID = output_dict[trial_type][0][trial_idx][0]
                trial_onset = np.round(output_dict[trial_type][0][trial_idx][1], _precision_decimals)
                trial_offset = np.round(output_dict[trial_type][0][trial_idx][2], _precision_decimals)
                resp_latency = np.round(output_dict[trial_type][0][trial_idx][3], _precision_decimals)

                response_auc_dff = np.round(output_dict[trial_type][1][trial_idx], _precision_decimals)
                response_peak_dff = np.round(output_dict[trial_type][2][trial_idx], _precision_decimals)
                baseline_auc_dff = np.round(output_dict[trial_type][3][trial_idx], _precision_decimals)
                response_auc_zscore = np.round(output_dict[trial_type][4][trial_idx], _precision_decimals)
                response_peak_zscore = np.round(output_dict[trial_type][5][trial_idx], _precision_decimals)
                baseline_auc_zscore = np.round(output_dict[trial_type][6][trial_idx], _precision_decimals)
                writer.writerow([subj_date] +
                                [trial_type] +
                                [trialID] +
                                [trial_onset] +
                                [trial_offset] +
                                [resp_latency] +
                                [response_auc_dff] +
                                [response_peak_dff] +
                                [baseline_auc_dff] +
                                [response_auc_zscore] +
                                [response_peak_zscore] +
                                [baseline_auc_zscore]
                                )

def output_json_to_flmm_format(data_dict, output_path, analysis_id: list):
    """Flatten a batch of session JSON dicts into a single wide CSV formatted for FLMM analysis.

    For every session, analysis ID, alignment, and trial type present,
    emits one output row per trial with trial metadata columns followed by
    one ``FP.<n>`` column per z-scored calcium signal timepoint. Handles
    slightly ragged per-trial signal lengths across sessions by truncating
    or NaN-padding to the running matrix width.

    Parameters
    ----------
    data_dict : list of dict
        Parsed session JSON dicts (e.g. from
        :func:`helpers.get_JSON_data.get_JSON_data`), each following the
        ``{'Subject', 'Date', 'AnalysisID': {...}}`` schema written by
        :func:`helpers.preprocess_files.preprocess_files` /
        :func:`measure_signals_and_save`.
    output_path : str
        Directory to write ``FLMM_FP_data.csv`` into.
    analysis_id : list
        Analysis IDs to include; each session's ``'AnalysisID'`` dict is
        indexed by every entry in this list.

    Returns
    -------
    None
        Side effect only: writes ``FLMM_FP_data.csv`` to ``output_path``.

    Raises
    ------
    ValueError
        If no trials were found across all of ``data_dict``/``analysis_id``
        (e.g. because the sessionData JSON has no populated trials for the
        given analysis ID) — raised with an explicit message instead of
        letting a bare ``IndexError`` surface from the CSV-header step below.

    Notes
    -----
    ``sig_matrix`` is built by repeated ``np.vstack`` calls inside the
    per-trial loop (O(n^2) in the number of trials); pre-allocating based on
    a known max trial count/length would be more efficient for large batches.
    """
    recording_list = list()
    analysisID_list = list()
    t_or_r_align_list = list()
    trialType_list = list()
    amdepth_list = list()
    trialID_list = list()
    trialOnset_list = list()
    respLatency_list = list()
    sig_matrix = np.array([])
    for session_data in data_dict:
        for analysisID in analysis_id:
            for t_or_r_align in session_data['AnalysisID'][analysisID]['Alignment']:
                for trialType in session_data['AnalysisID'][analysisID]['Alignment'][t_or_r_align]['Trial type']:
                    cur_trialType_data = session_data['AnalysisID'][analysisID]['Alignment'][t_or_r_align]['Trial type'][trialType]
                    if len(cur_trialType_data) == 0:
                        continue
                    n_trials = len(cur_trialType_data['TrialID'])

                    # Info
                    recording_list.extend([session_data['Subject'] + '_' + session_data['Date']] * n_trials)
                    analysisID_list.extend([analysisID] * n_trials)
                    t_or_r_align_list.extend([t_or_r_align] * n_trials)
                    trialType_list.extend([trialType] * n_trials)

                    # Behavioral data
                    amdepth_list.extend(cur_trialType_data['AMdepth'])
                    trialID_list.extend(cur_trialType_data['TrialID'])
                    trialOnset_list.extend(cur_trialType_data['Trial_onset'])
                    respLatency_list.extend(cur_trialType_data['RespLatency'])

                    # Calcium data
                    for trial_idx in range(n_trials):
                        cur_trial = np.array(cur_trialType_data['Calcium_zscore'][trial_idx])
                        if sig_matrix.size == 0:
                            sig_matrix = np.hstack((sig_matrix, cur_trial))
                        else:
                            try:
                                sig_matrix = np.vstack((sig_matrix, cur_trial))
                            except ValueError:  # Different sessions might have slightly different number of points
                                try:
                                    sig_matrix = np.vstack((sig_matrix, cur_trial[0:sig_matrix.shape[1]]))  # new array is larger
                                except ValueError:
                                    nan_to_fill = np.full(sig_matrix.shape[1] - cur_trial.size, np.nan)
                                    cur_trial = np.append(cur_trial, nan_to_fill)
                                    sig_matrix = np.vstack((sig_matrix, cur_trial))

    if len(recording_list) == 0:
        raise ValueError(
            'No trials were found for analysis_id=' + str(analysis_id) +
            ' in the provided data_dict. Check that the sessionData JSON files '
            "actually have populated 'Trial type' entries for this analysis ID "
            "(e.g. that extract_trial_zscores and output_sessionData_json were "
            'enabled when the pipeline was run), and that analysis_id matches '
            "an 'AnalysisID' key present in the JSON.")

    # sig_matrix is only guaranteed to be 2-D once a second trial has been
    # vstack'd onto it (the very first trial is added via hstack, which stays
    # 1-D); promote it here so a single-trial batch doesn't crash below.
    sig_matrix = np.atleast_2d(sig_matrix)

    # Output CSV
    file_name = 'FLMM_FP_data'
    photometry_columns_prefix = 'FP.'
    with open(sep.join([output_path, file_name + '.csv']), 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        csv_header = ['Recording', 'AnalysisID', 'Alignment', 'Trial_type',
                      'AMDepth', 'TrialID', 'Trial_onset', 'RespLatency']
        csv_header.extend([photometry_columns_prefix + str(dummy_idx + 1) for dummy_idx in range(sig_matrix.shape[1])])

        writer.writerow(csv_header)

        for trial_idx in range(len(recording_list)):
            csv_row = [recording_list[trial_idx], analysisID_list[trial_idx], t_or_r_align_list[trial_idx], trialType_list[trial_idx],
                       amdepth_list[trial_idx], trialID_list[trial_idx], trialOnset_list[trial_idx], respLatency_list[trial_idx],
                       *sig_matrix[trial_idx, :]]
            writer.writerow(csv_row)
