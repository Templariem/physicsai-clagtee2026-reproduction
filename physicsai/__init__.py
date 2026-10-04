"""Fixed-protocol reproduction of the CLAGTEE 2026 paper."""
import os
from pathlib import Path
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parents[1]/'.cache/matplotlib'))
