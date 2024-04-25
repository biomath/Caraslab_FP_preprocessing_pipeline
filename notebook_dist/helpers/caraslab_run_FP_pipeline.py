from platform import system
from os.path import sep

from notebook_dist.helpers.extract_FP_trial_zscores import run_zscore_extraction
from notebook_dist.helpers.recalculate_ePsych_responseLatency import recalculate_ePsych_responseLatency

# Tweak the regex file separator for cross-platform compatibility
if system() == 'Windows':
    REGEX_SEP = sep * 2
else:
    REGEX_SEP = sep


def run_pipeline(input_list):
    # Gather settings
    _, SETTINGS_DICT = input_list
    pipeline_switchboard = SETTINGS_DICT['PIPELINE_SWITCHBOARD']

    if pipeline_switchboard['recalculate_ePsych_responseLatency'] and SETTINGS_DICT['EXPERIMENT_TYPE'] == 'AversiveAM':
        recalculate_ePsych_responseLatency(input_list)

    if pipeline_switchboard['extract_trial_zscores'] or pipeline_switchboard['plot_AMDepth_zscores']:
        run_zscore_extraction(input_list)