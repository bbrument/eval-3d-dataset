"""Pytest configuration for eval_pipeline tests."""

import sys
from pathlib import Path

# F4 Fix: Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
