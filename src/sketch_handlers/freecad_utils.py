import sys
import os
import typing

def ensure_freecad_imported(libpath: str = "~/github/FreeCAD/build/lib") -> typing.Tuple[typing.Any, typing.Any, typing.Any]:
    """
    Safely imports and returns FreeCAD modules (FreeCAD, Part, Sketcher).
    Ensures the path is set up correctly before importing.

    Returns:
        tuple: (FreeCAD, Part, Sketcher) modules.

    Raises:
        ImportError: If FreeCAD modules cannot be found or imported.
    """
    # 1. Ensure FreeCAD path is in sys.path
    # This is typically handled in __init__.py, but safe to double-check here
    freecad_lib = os.path.expanduser(libpath)
    if freecad_lib not in sys.path:
        sys.path.insert(0, freecad_lib)

    try:
        import FreeCAD
        import Part
        import Sketcher
        return FreeCAD, Part, Sketcher
    except ImportError as e:
        raise ImportError(f"Failed to import FreeCAD modules: {e}. "
                          "Ensure you are running in an environment with FreeCAD available "
                          "or that the path is configured correctly.") from e
