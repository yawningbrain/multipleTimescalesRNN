"""Small helpers kept for backward compatibility with the original notebooks.

New code should use :mod:`rnn_library.network` and :mod:`rnn_library.analysis`.
"""

from .analysis import half_width_at_half_max


def _ask(cast, error_message):
    while True:
        try:
            result = cast(input())
        except ValueError:
            print(f"\t{error_message} Please try again!")
            continue
        print("\tEntry accepted")
        return result


def ask_user_integer():
    """Prompt on stdin until the user enters an integer."""
    return _ask(int, "The entry is not an integer.")


def ask_user_float():
    """Prompt on stdin until the user enters a number."""
    return _ask(float, "The entry is not a number.")


def FWHM_funct(time_list, autocorrelation, start=0, end=None):
    """Half width at half maximum of an ACF (legacy name; returns ``nan`` on failure).

    Equivalent to :func:`rnn_library.analysis.half_width_at_half_max` applied
    to ``[start:end]``.
    """
    return half_width_at_half_max(time_list[start:end], autocorrelation[start:end])
