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
    _, SETTINGS_DICT, run_type = input_list
    input_list = (input_list[0], input_list[1])

    pipeline_switchboard = SETTINGS_DICT['PIPELINE_SWITCHBOARD']
    SETTINGS_DICT['TRIAL_OR_RESPONSE_ALIGNED'] = run_type  # Just so I don't need to update it in all functions

    if pipeline_switchboard['extract_trial_zscores'] or pipeline_switchboard['plot_AMDepth_zscores']:
        run_zscore_extraction(input_list)
    
    if pipeline_switchboard['extract_extinction_spoutOff_zscores'] or pipeline_switchboard['plot_extinction_spoutOff_zscores']:
        run_zscore_extraction(input_list)