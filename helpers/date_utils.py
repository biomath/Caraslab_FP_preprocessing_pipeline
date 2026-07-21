"""Extracts the date and timestamp embedded in Caraslab experiment tags/filenames, regardless of their position.

Recording filenames/experiment tags follow the convention
``<...>-<YYMMDD>-<HHMMSS><...>`` (e.g. ``FP-Aversive-AM-220119-111243``), but
the number of words preceding the date varies across experiments/paradigms.
These functions locate the date/timestamp by content (a standalone 6-digit
token) rather than by a fixed, experiment-specific position index.
"""
from re import search, findall


def find_date_in_tag(tag):
    """Find the date (YYMMDD) in an experiment tag, regardless of its position.

    Looks for the first standalone 6-digit token (e.g. 'FP-Aversive-AM-220119-111243' -> '220119').
    The timestamp that follows the date is also 6 digits, but since it always comes
    *after* the date in this naming convention, searching left-to-right finds the date first.

    Parameters
    ----------
    tag : str
        Experiment tag or filename segment containing the date (and usually
        a following timestamp), e.g. ``'FP-Aversive-AM-220119-111243'``.

    Returns
    -------
    str
        The 6-character YYMMDD date substring.

    Raises
    ------
    ValueError
        If no standalone 6-digit token is found in ``tag``.
    """
    date_match = search(r'(?<!\d)\d{6}(?!\d)', tag)
    if date_match is None:
        raise ValueError(f"Could not find a 6-digit (YYMMDD) date in experiment tag: '{tag}'")
    return date_match.group(0)


def find_timestamp_in_tag(tag):
    """Find the recording timestamp (HHMMSS) in an experiment tag, regardless of its position.

    The timestamp is the second standalone 6-digit token in the tag,
    immediately following the date (see :func:`find_date_in_tag`) in this
    naming convention (e.g. ``'FP-Aversive-AM-220119-111243'`` -> ``'111243'``).

    Parameters
    ----------
    tag : str
        Experiment tag or filename segment containing the date and timestamp,
        e.g. ``'FP-Aversive-AM-220119-111243'``.

    Returns
    -------
    str
        The 6-character HHMMSS timestamp substring.

    Raises
    ------
    ValueError
        If fewer than two standalone 6-digit tokens are found in ``tag``.
    """
    matches = findall(r'(?<!\d)\d{6}(?!\d)', tag)
    if len(matches) < 2:
        raise ValueError(
            f"Could not find a 6-digit (HHMMSS) timestamp following the date in experiment tag: '{tag}'")
    return matches[1]
