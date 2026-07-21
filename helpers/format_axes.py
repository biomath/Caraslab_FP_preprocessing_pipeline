"""Shared matplotlib axis-formatting helper used across the plotting modules."""


def format_ax(ax, **kwargs):
    """Apply the pipeline's standard axis styling to a matplotlib Axes.

    Hides the top/right spines and restricts ticks to the left/bottom
    spines. Optionally sets axis labels.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes instance to style in place.
    **kwargs
        Only ``'xlabel'`` and ``'ylabel'`` (both ``str``) are recognized and
        applied via ``ax.set_xlabel``/``ax.set_ylabel``; any other keyword is
        silently ignored (a typo in a kwarg name will not raise an error).

    Returns
    -------
    None
    """
    kwargs_dict = dict(kwargs)
    if 'ylabel' in kwargs_dict:
        ax.set_ylabel(kwargs_dict['ylabel'])
    if 'xlabel' in kwargs_dict:
        ax.set_xlabel(kwargs_dict['xlabel'])
    # ax.tick_params(axis='both', which='major')
    # ax.axhline(y=0, linestyle=':', color='k')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    # ax.xaxis.get_major_ticks()[-1].label1.set_visible(False)
    # ax.xaxis.get_major_ticks()[0].label1.set_visible(False)