from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING
import numpy as np
import sketchgraphs.data as datalib
from sketchgraphs.data._constraint import ConstraintParameterType, QuantityParameter
from ..freecad_utils import ensure_freecad_imported

if TYPE_CHECKING:
    FreeCAD, Part, Sketcher = ensure_freecad_imported()


class ConstraintHandler(ABC):
    """
    Base class for constraint handlers.
    Each handler is responsible for a single constraint type (e.g. Coincident, Distance).
    """

    @property
    @abstractmethod
    def token_id(self) -> int:
        """The unique token ID for this constraint type."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """The name of this constraint type (must match ConstraintType value)."""
        pass

    @abstractmethod
    def encode(self, constraint: Any, primitive_tokenpos: dict[str, int], entities: dict[str, datalib.Entity]) -> tuple[int, tuple[int, ...], np.ndarray, tuple[int, ...]]:
        """
        Encode a constraint into token ID, reference tokens, and parameter values.

        Args:
            constraint: The constraint object.
            primitive_tokenpos: A dictionary mapping Entity ID (str) -> Start Token Index (int).
            entities: A dictionary mapping Entity ID (str) -> Entity object (datalib.Entity).

        Returns:
            token_id: The constraint type token ID.
            ref_tokens: A tuple of reference tokens (pointers to primitive tokens).
            params: A numpy array of parameter values (if any).
            param_tokens: A tuple of parameter token IDs corresponding to params.
        """
        pass

    @abstractmethod
    def decode(self, ref_indices: tuple[int, ...], params: np.ndarray, tokenpos_to_primitive: dict[int, str]) -> datalib.Constraint:
        """
        Reconstruct a constraint from reference indices and parameters.

        Args:
            ref_indices: Tuple of token positions referenced by this constraint.
            params: Array of parameter values.
            tokenpos_to_primitive: Dictionary mapping token position -> Entity ID (or Subnode ID).

        Returns:
            The reconstructed Constraint object.
        """
        pass

    @abstractmethod
    def to_freecad(self, constraint: datalib.Constraint, primitive_geoID: dict[str, tuple[int, int]]) -> list[tuple[str, list, float | None]]:
        """
        Convert a SketchGraphs Constraint to FreeCAD constraint argument tuples.

        Args:
            constraint: The SketchGraphs Constraint object.
            primitive_geoID: Dictionary mapping reference string (e.g. "0", "0.start", "0.center")
                             -> (GeoID, PointID) tuple.

        Returns:
            List of (fc_constraint_type, args_list) tuples.
            Each tuple is used as: Sketcher.Constraint(fc_constraint_type, *args_list)
        """
        pass

    @abstractmethod
    def from_freecad(self, fc_constraint: 'Sketcher.Constraint', geoID_to_primitive: dict[tuple[int, int], str]) -> datalib.Constraint | None:
        """
        Convert a FreeCAD constraint to a SketchGraphs Constraint.

        Args:
            fc_constraint: The FreeCAD Constraint object.
            geoID_to_primitive: Dictionary mapping (GeoID, PointID) tuple -> reference string
                                (e.g. "0", "0.start", "0.center").
        """
        pass

    def get_constraint_value(self, constraint: datalib.Constraint) -> float:
        """Helper to extract value from constraint parameters."""
        for param in constraint.parameters:
            if param.type == ConstraintParameterType.Quantity:
                if isinstance(param, QuantityParameter):
                    # value might be string or float
                    try:
                        return float(param.value)
                    except (ValueError, TypeError):
                        return 0.0
        return 0.0

    # Types that don't exist in SketchGraphs ConstraintType -> map to nearest equivalent
    _SG_TYPE_MAP = {
        'DistanceX': 'Distance',
        'DistanceY': 'Distance',
        'Symmetric': 'Mirror',
    }

    def create_constraint(self, constraint_type_name: str, refs: tuple[str, ...], value: float | None = None) -> datalib.Constraint:
        """Helper to create a sketchgraphs.data.Constraint object."""
        from sketchgraphs.data._constraint import Constraint, ConstraintType, LocalReferenceParameter, QuantityParameter

        # Map to valid SketchGraphs enum; keep original name as identifier
        sg_type_name = self._SG_TYPE_MAP.get(constraint_type_name, constraint_type_name)
        ctype = getattr(ConstraintType, sg_type_name)
        params = []
        for i, ref in enumerate(refs):
            params.append(LocalReferenceParameter(parameterId=f"local{i}", value=ref))

        if value is not None:
            pid = 'angle' if constraint_type_name == 'Angle' else 'length'
            params.append(QuantityParameter(parameterId=pid, value=value, expression=str(value)))

        return Constraint(identifier=constraint_type_name, constraint_type=ctype, parameters=params)

    def _parse_references(self, constraint: Any, primitive_tokenpos: dict[str, int]) -> list[tuple[str, str | None]]:
        """
        Parse and filter references from a constraint.
        Returns a list of (entity_id, subnode_name) tuples.
        """
        refs = []
        for param in constraint.parameters:
            if type(param).__name__ != 'LocalReferenceParameter': continue

            value = getattr(param, 'value', None)
            if not isinstance(value, str): continue

            if '.' in value:
                eid, sub = value.split('.', 1)
                if sub not in ('start', 'end', 'center'): continue
                refs.append((eid, sub))
            else:
                refs.append((value, None))

        # Filter external refs
        filtered_refs = []
        for eid, sub in refs:
            key = f"{eid}.{sub}" if sub else eid
            if key in primitive_tokenpos:
                filtered_refs.append((eid, sub))
            else:
                return [] # Invalidate if any ref is missing

        return filtered_refs

    def _get_sorted_ref_tokens(self, constraint: Any, primitive_tokenpos: dict[str, int], entities: dict[str, datalib.Entity]) -> tuple[int, ...]:
        """
        Helper to parse, sort, and convert references to tokens.
        """
        refs = self._parse_references(constraint, primitive_tokenpos)
        refs = list(set(refs))

        # Sort references by token position
        def get_ref_pos(r):
            key = f"{r[0]}.{r[1]}" if r[1] else r[0]
            return primitive_tokenpos.get(key, 0)

        sorted_refs = sorted(list(set(refs)), key=get_ref_pos)

        ref_tokens = []
        for eid, sub in sorted_refs:
            key = f"{eid}.{sub}" if sub else eid
            ref_tokens.append(primitive_tokenpos[key])
        return tuple(ref_tokens)

    def _get_ref_strings(self, constraint: datalib.Constraint) -> tuple[str, ...]:
        """
        Extract reference strings from constraint parameters.
        Returns list of reference strings like ["0.start", "1", "2.center"].
        """
        refs = []
        for param in constraint.parameters:
            if type(param).__name__ != 'LocalReferenceParameter':
                continue
            value = getattr(param, 'value', None)
            if isinstance(value, str):
                refs.append(value)
        return tuple(refs)

    def validate_refs(self, refs: tuple[int | str, ...]) -> bool:
        """Validate if the number of references is correct for this constraint."""
        return True # Default lenient
