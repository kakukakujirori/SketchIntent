import os
import sys

# 1. Setup FreeCAD Paths (Centralized)
freecad_lib = os.path.expanduser("~/github/FreeCAD/build/lib")
if freecad_lib not in sys.path:
    sys.path.insert(0, freecad_lib)

# 2. FreeCAD check moved to individual modules or lazy check if needed
FREECAD_AVAILABLE = False
try:
    import FreeCAD
    FREECAD_AVAILABLE = True
except ImportError:
    pass

from .context import SketchContext
from .tokenizer import SketchTokenizer
from .constants import PrimitiveToken, ConstraintToken

__all__ = ['SketchContext', 'SketchTokenizer', 'PrimitiveToken', 'ConstraintToken', 'FREECAD_AVAILABLE']
