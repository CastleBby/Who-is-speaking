"""
temporal_features.py

Overview:
helper functions for transforming the frame level mouth open
ratio into time-based features. These temporal features are more useful because speech is 
not a single frame event.

USAGE:
Example:
    df["mouth_change"] = compute_frame_to_frame_change(df["mouth_ratio"])
    df["mouth_ratio_smooth"] = compute_rolling_mean(df["mouth_ratio"], window=5)
    df["speaking_score"] = compute_motion_energy(df["mouth_ratio"], window=5)

Inputs:
- A pandas Series containing mouth-open ratio values ordered over time.

Outputs:
- A pandas Series containing a derived temporal feature.

"""

import pandas as pd


def compute_frame_to_frame_change(signal: pd.Series) -> pd.Series:
    """
    Compute the absolute frame-to-frame change of a signal

    Returns
    -------
    pd.Series
        The absolute difference between consecutive values.
        The first value is filled with 0.0 because there is no previous frame
    """

    # Compute the difference between each value and the one before it.
    diff = signal.diff()

    # Take the absolute value so that both upward and downward changes count
    # as movement.
    abs_diff = diff.abs()

    # Replace the first missing value with 0.0 because there is no previous
    # frame to compare against.
    return abs_diff.fillna(0.0)


def compute_rolling_mean(signal: pd.Series, window: int = 5) -> pd.Series:
    """
    Compute a rolling mean to smooth a signal over time.

    Purpose:
    This reduces short-term jitter and makes overall patterns easier to see.

    Parameters
    ----------
    signal : pd.Series
        A time-ordered numeric signal.
    window : int, optional
        Number of frames to include in the moving window. Default is 5.

    Returns
    -------
    pd.Series
        A smoothed version of the input signal.
    """

    # Compute the moving average over the specified window.
    # min_periods=1 ensures the first few rows still receive values.
    return signal.rolling(window=window, min_periods=1).mean()


def compute_motion_energy(signal: pd.Series, window: int = 5) -> pd.Series:
    """
    Compute a rolling mouth-motion score from a raw signal

    Returns
    -------
    pd.Series
        A rolling motion-energy signal
    """

    # First compute how much the signal changes from frame to frame.
    frame_change = compute_frame_to_frame_change(signal)

    # Then smooth those changes over a short time window to estimate
    # sustained motion rather than one-frame spikes.
    motion_energy = frame_change.rolling(window=window, min_periods=1).mean()

    return motion_energy