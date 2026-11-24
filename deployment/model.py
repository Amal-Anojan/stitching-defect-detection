"""Compatibility import for existing deployment integrations."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from seam_defect.models import FPNClassifier
