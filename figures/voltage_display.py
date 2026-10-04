"""One numerical voltage-to-color mapping for Figures 4(a) and 5.

Author-approved range on 2026-10-01. Latent visualizations are excluded.
"""
import matplotlib
from matplotlib.colors import Normalize

VOLTAGE_MIN = 0.1
VOLTAGE_MAX = 8.0
VOLTAGE_TICKS = [0.1, 1, 2, 3, 4, 5, 6, 7, 8]
VOLTAGE_NORM = Normalize(VOLTAGE_MIN, VOLTAGE_MAX, clip=True)
VOLTAGE_CMAP = matplotlib.colormaps['turbo']
