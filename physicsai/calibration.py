"""Calibration from the five author-confirmed LED pairs in Figure 3 only."""
import numpy as np

COLORS = ('white', 'blue', 'green', 'yellow', 'red')
DISTANCE_CM = np.array([5.0, 6.0, 6.5, 12.0, 17.0])
VOLTAGE_V = np.array([3.3, 3.1, 2.5, 2.1, 2.0])
_slope, _intercept = np.polyfit(np.log(DISTANCE_CM), np.log(VOLTAGE_V), 1)
A_V = float(np.exp(_intercept))
GAMMA = float(-_slope)


def voltage(distance_cm):
    d = np.asarray(distance_cm, dtype=float)
    # The power law is defined only at positive distances. Missing locations
    # remain NaN and are excluded before fitting or scoring.
    return A_V * np.power(np.where(d > 0, d, np.nan), -GAMMA)


def decay_derivative(distance_cm):
    d = np.asarray(distance_cm, dtype=float)
    return -(voltage(d + 0.05) - voltage(d - 0.05)) / 0.10


def within_calibration_interval(distance_cm):
    d = np.asarray(distance_cm, dtype=float)
    return (d >= DISTANCE_CM.min()) & (d <= DISTANCE_CM.max())
