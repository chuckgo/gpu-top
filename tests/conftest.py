"""Add project root to sys.path so tests can import gpu_top."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
