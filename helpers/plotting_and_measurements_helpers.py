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
                                 paradigm_type, response_latency_filter, response_window_duration,
                                 min_length, ams_to_analyze, baseline_start_for_zscore, _precision_decimals,
                                 all_hit_color, hitShock_color, hitNoShock_color, miss_color, missShock_color,
                                 missNoShock_color, fa_color, reject_color, passive_color):
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
                ax.axvspan(response_latency_filter, response_window_duration,
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
            else:  # Passive
                cur_color = passive_color

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

            if trial_type == 'Miss (no shock)' or trial_type == 'Hit (no shock)':
                linestyle = '--'
            else:
                linestyle = '-'



            signals_mean = np.nanmean(plot_sigs, axis=0)
            signals_sem = np.nanstd(plot_sigs, axis=0, ddof=1) / np.sqrt(
                np.count_nonzero(~np.isnan(plot_sigs), axis=0))
            x_axis = np.linspace(-baseline_start_for_zscore, response_window_duration, len(signals_mean))
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
                                   paradigm_type, response_latency_filter, response_window_duration,
                                   min_length, ams_to_analyze, baseline_start_for_zscore, _precision_decimals):
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
                            'False alarm']
        elif paradigm_type == '1IFC':
            trial_groups = ['Hit', 'Reject', 'Miss', 'False alarm']
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
                    ax.axvspan(response_latency_filter, response_window_duration,
                               ymin=0.025, ymax=0.05, facecolor='g', alpha=0.25)

            # will just be each trial type for aversive
            cur_trialTypes = list(ttype for ttype in trial_type_dict.keys() if ttype in tgroup)
            legend_handles = list()
            for trial_type in cur_trialTypes:
                am_sig_list = dict()

                if trial_type == 'Miss (no shock)' or trial_type == 'Hit (no shock)':
                    linestyle = '--'
                else:
                    linestyle = '-'

                for amdepth in sorted(list(set(all_ams)), reverse=True):
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
                    x_axis = np.linspace(-baseline_start_for_zscore, response_window_duration,
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


def __calculate_PeakValue_and_AUC(sigs, trial_info, baseline_window_start_time,
                                  sampling_frequency, x_axis,
                                  auc_start=0, auc_end=4):
    auc_response = np.zeros(np.shape(sigs)[0])
    peak = np.zeros(np.shape(sigs)[0])
    auc_baseline = np.zeros(np.shape(sigs)[0])
    for trial_idx, cur_trial in enumerate(trial_info):
        bounded_response_xaxis = x_axis[int((auc_start + baseline_window_start_time) * sampling_frequency):
                                        int((auc_end + baseline_window_start_time) * sampling_frequency)]

        bounded_response = sigs[trial_idx, int((auc_start + baseline_window_start_time) * sampling_frequency):
                                           int((auc_end + baseline_window_start_time) * sampling_frequency)]

        # 0-index is already baseline_start_time
        bounded_baseline_xaxis = x_axis[0:int(baseline_window_start_time * sampling_frequency)]
        bounded_baseline = sigs[trial_idx, 0:int(baseline_window_start_time * sampling_frequency)]

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
                             output_plots_path, response_window_duration,
                             min_length, ams_to_analyze, baseline_start_for_zscore, _precision_decimals,
                             sampling_frequency, auc_start, auc_end, output_path, output_sessionData_json):
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
                    continue
            else:
                dff_sigs[i, 0:len(ts_dff)] = ts_dff
                zscore_sigs[i, 0:len(ts_zscore)] = ts_zscore

        if np.size(dff_sigs) == 0 or np.size(zscore_sigs) == 0:
            continue

        # Measure and add measurements to list
        trial_info = trial_type_dict[trial_type]['info']
        x_axis = np.linspace(-baseline_start_for_zscore, response_window_duration,
                             np.shape(zscore_sigs)[1])  # just in case

        # Get the dff signal measurements
        auc_response_dff, peak_dff, auc_baseline_dff = __calculate_PeakValue_and_AUC(
            dff_sigs, trial_info, baseline_window_start_time=baseline_start_for_zscore,
            sampling_frequency=sampling_frequency, x_axis=x_axis, auc_start=auc_start, auc_end=auc_end)

        # Get the z-scored signal measurements
        auc_response_zscore, peak_zscore, auc_baseline_zscore = __calculate_PeakValue_and_AUC(
            zscore_sigs, trial_info, baseline_window_start_time=baseline_start_for_zscore,
            sampling_frequency=sampling_frequency, x_axis=x_axis, auc_start=auc_start, auc_end=auc_end)

        output_dict.update({trial_type: (trial_info,
                                         auc_response_dff,
                                         peak_dff,
                                         auc_baseline_dff,
                                         auc_response_zscore,
                                         peak_zscore,
                                         auc_baseline_zscore)})

        # Output individual session info, curves and measurements in json files here
        if output_sessionData_json:
            for trial_type in output_dict.keys():
                # Trial info
                cur_sessionData['AnalysisID'][analysis_id]['Alignment'][t_or_r_align]['Trial type'][trial_type][
                    'TrialID'] = \
                    [x[0] for x in output_dict[trial_type][0]]
                cur_sessionData['AnalysisID'][analysis_id]['Alignment'][t_or_r_align]['Trial type'][trial_type][
                    'AMdepth'] = \
                    [np.round(x[1], 2) for x in output_dict[trial_type][0]]
                cur_sessionData['AnalysisID'][analysis_id]['Alignment'][t_or_r_align]['Trial type'][trial_type][
                    'Trial_onset'] = \
                    [np.round(x[2], _precision_decimals) for x in output_dict[trial_type][0]]
                cur_sessionData['AnalysisID'][analysis_id]['Alignment'][t_or_r_align]['Trial type'][trial_type][
                    'Trial_offset'] = \
                    [np.round(x[3], _precision_decimals) for x in output_dict[trial_type][0]]
                cur_sessionData['AnalysisID'][analysis_id]['Alignment'][t_or_r_align]['Trial type'][trial_type][
                    'RespLatency'] = \
                    [np.round(x[4], _precision_decimals) for x in output_dict[trial_type][0]]

                # Signal measurements
                cur_sessionData['AnalysisID'][analysis_id]['Alignment'][t_or_r_align]['Trial type'][trial_type][
                    'Response_auc_dff'] = (
                    np.round(output_dict[trial_type][1], _precision_decimals))
                cur_sessionData['AnalysisID'][analysis_id]['Alignment'][t_or_r_align]['Trial type'][trial_type][
                    'Response_peak_dff'] = (
                    np.round(output_dict[trial_type][2], _precision_decimals))
                cur_sessionData['AnalysisID'][analysis_id]['Alignment'][t_or_r_align]['Trial type'][trial_type][
                    'Baseline_auc_dff'] = (
                    np.round(output_dict[trial_type][3], _precision_decimals))
                cur_sessionData['AnalysisID'][analysis_id]['Alignment'][t_or_r_align]['Trial type'][trial_type][
                    'Response_auc_zscore'] = (
                    np.round(output_dict[trial_type][4], _precision_decimals))
                cur_sessionData['AnalysisID'][analysis_id]['Alignment'][t_or_r_align]['Trial type'][trial_type][
                    'Response_peak_zscore'] = (
                    np.round(output_dict[trial_type][5], _precision_decimals))
                cur_sessionData['AnalysisID'][analysis_id]['Alignment'][t_or_r_align]['Trial type'][trial_type][
                    'Baseline_auc_zscore'] = (
                    np.round(output_dict[trial_type][6], _precision_decimals))

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

        writer.writerow(['Recording'] + ['Trial_type'] + ['TrialID'] + ['AMDepth'] +
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

                # Reference:
                # writer.writerow(['Recording'] + ['Trial_type'] + ['TrialID'] + ['AMDepth'] +
                #                             ['Trial_Onset'] +
                #                             ['Trial_Offset'] +
                #                             ['RespLatency'] +
                #                             ['Area_under_curve_dff'] +
                #                             ['Peak_value_dff'] +
                #                             ['Baseline_area_under_curve_dff'] +
                #                             ['Area_under_curve_zscore'] +
                #                             ['Peak_value_zscore'] +
                #                             ['Baseline_area_under_curve_zscore'])
                writer.writerow([subj_date] + [trial_type] + [trialID] + [AMdepth] +
                                [np.round(trial_onset, _precision_decimals)] +  # Trial onset
                                [np.round(trial_offset, _precision_decimals)] +
                                [np.round(resp_latency, _precision_decimals)] +
                                [np.round(output_dict[trial_type][1][trial_idx],
                                          _precision_decimals)] +  # Response AUC dff
                                [np.round(output_dict[trial_type][2][trial_idx],
                                          _precision_decimals)] +  # Response Peak dff
                                [np.round(output_dict[trial_type][3][trial_idx],
                                          _precision_decimals)] +  # Baseline AUC dff
                                [np.round(output_dict[trial_type][4][trial_idx],
                                          _precision_decimals)] +  # Response AUC zscore
                                [np.round(output_dict[trial_type][5][trial_idx],
                                          _precision_decimals)] +  # Response Peak zscore
                                [np.round(output_dict[trial_type][6][trial_idx], _precision_decimals)]
                                # Baseline AUC zscore
                                )
