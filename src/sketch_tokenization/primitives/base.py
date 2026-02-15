from abc import ABC, abstractmethod
from typing import Any
import numpy as np

class PrimitiveHandler(ABC):
    """
    Base class for primitive handlers.
    Each handler is responsible for a single primitive type (e.g. Line, Circle).
    """

    @property
    @abstractmethod
    def token_id(self) -> int:
        """The unique token ID for this primitive type."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """The name of this primitive type (must match NodeOp.label)."""
        pass

    @property
    @abstractmethod
    def layout(self) -> tuple[int, ...]:
        """Sequence of PrimitiveParamTokens defining the output structure."""
        pass

    @property
    def n_params(self) -> int:
        """Number of parameters, derived from layout."""
        return len(self.layout)

    def get_subnode_offset(self) -> dict[str, int]:
        """
        Derive subnode offsets relative to start_pos (Type token).
        Parameters start at offset +1.
        """
        mapping = {}
        # Import here to avoid circular dependency
        from ..constants import PrimitiveAttrToken

        # Subnodes are usually 2D points (x, y). We map the name to the index of the first coordinate.
        # e.g. "start" -> index of StartX

        for i, token_type in enumerate(self.layout):
            # In token sequence: [Type, Param0, Param1, ...]
            # So +1 is necessary for the Type token.
            offset = 1 + i

            # Only check the x coordinate
            if token_type == PrimitiveAttrToken.StartX:
                mapping["start"] = offset
            elif token_type == PrimitiveAttrToken.EndX:
                mapping["end"] = offset
            elif token_type == PrimitiveAttrToken.CenterX:
                mapping["center"] = offset
            elif token_type == PrimitiveAttrToken.MidX:
                mapping["mid"] = offset

        return mapping

    @abstractmethod
    def encode(self, entity: Any) -> tuple[np.ndarray, tuple[int, ...]]:
        """
        Encode a primitive entity into numerical parameters and coordinate tokens.

        Args:
            entity: The primitive entity (e.g. datalib.Line)

        Returns:
            values: A numpy array of parameter values (e.g. [x1, y1, x2, y2])
            coord_tokens: A tuple of coordinate token IDs corresponding to the values.
        """
        pass

    @abstractmethod
    def decode(self, params: np.ndarray, entity_id: str) -> Any:
        """
        Decode numerical parameters into a primitive entity.

        Args:
            params: A numpy array of parameter values.
            entity_id: The ID to assign to the new entity.

        Returns:
            entity: The reconstructed primitive entity.
        """
        pass

    @abstractmethod
    def to_freecad(self, entity: Any, sketch: Any, **kwargs) -> Any:
        """
        Create the corresponding FreeCAD geometry in the given sketch.

        Args:
            entity: The primitive entity (e.g. datalib.Line)
            sketch: The FreeCAD SketchObject to add geometry to.
            **kwargs: Additional arguments if needed.

        Returns:
            The created FreeCAD Geometry object or ID.
        """
        pass

    @abstractmethod
    def from_freecad(self, fc_geo: Any, **kwargs) -> Any:
        """
        Create a primitive entity from a FreeCAD geometry object.

        Args:
            fc_geo: The FreeCAD Geometry object.
            **kwargs: Additional context if needed.

        Returns:
            The primitive entity (e.g. datalib.Line).
        """
        pass

    @abstractmethod
    def get_bbox(self, entity: Any) -> np.ndarray:
        """Calculate the bounding box of the entity."""
        pass
