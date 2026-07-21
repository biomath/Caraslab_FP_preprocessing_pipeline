"""Canonical JSON encoder that knows how to serialize numpy scalar and array types."""
import json
from numpy import integer, floating, ndarray


class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types."""

    def default(self, obj):
        """Convert a numpy value to a plain Python/JSON-serializable equivalent.

        Parameters
        ----------
        obj : object
            Object being serialized by :func:`json.dumps` that could not be
            handled by the default encoder (e.g. a numpy scalar or array).

        Returns
        -------
        int or float or list or Any
            ``obj`` converted to a native Python ``int``, ``float``, or
            ``list`` if it is a numpy integer, floating-point, or ndarray
            instance, respectively. Otherwise, falls back to
            ``json.JSONEncoder.default``, which raises a ``TypeError`` for
            still-unsupported types.
        """
        if isinstance(obj, integer):
            return int(obj)
        elif isinstance(obj, floating):
            return float(obj)
        elif isinstance(obj, ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)