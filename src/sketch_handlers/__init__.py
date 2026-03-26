from .constants import PrimitiveToken, ConstraintToken
from .context import SketchContext
from .sketch_freecad_converter import SketchFreecadConverter
from .sketch_tokenizer import SketchTokenizer

# IMPORTANT: Importing these packages ensures that their internal handlers
# are automatically registered into PRIMITIVE_HANDLERS and CONSTRAINT_HANDLERS
# via their @register_module() decorators upon any import of sketch_handlers.
# This is required so standalone utils (e.g. normalize_sketch) can confidently use handlers.
from . import primitives
from . import constraints

__all__ = ['SketchContext', 'SketchTokenizer', 'PrimitiveToken', 'ConstraintToken', 'FREECAD_AVAILABLE']
