from platform import system
from os.path import sep

from helpers.extract_FP_trial_zscores import run_zscore_extraction

# Tweak the regex file separator for cross-platform compatibility
if system() == 'Windows':
    REGEX_SEP = sep * 2
else:
    REGEX_SEP = sep


def run_pipeline(input_list):
    # Gather settings
    _, SETTINGS_DICT = input_list
    pipeline_switchboard = SETTINGS_DICT['PIPELINE_SWITCHBOARD']

    if pipeline_switchboard['extract_trial_zscores'] or pipeline_switchboard['plot_AMDepth_zscores']:
        run_zscore_extraction(input_list)
    
    if pipeline_switchboard['extract_extinction_spoutOff_zscores'] or pipeline_switchboard['plot_extinction_spoutOff_zscores']:
        run_zscore_extraction(input_list)