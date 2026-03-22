from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING
import numpy as np
import sketchgraphs.data as datalib
from sketchgraphs.data._constraint import (
    Constraint,
    ConstraintType,
    ConstraintParameterType,
    DirectionValue,
    BooleanParameter,
    EnumParameter,
    LocalReferenceParameter,
    QuantityParameter,
)
from src.sketch_handlers.freecad_utils import ensure_freecad_imported

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
    def encode(self, constraint: Any, primitive_tokenpos: dict[str, int]) -> tuple[int, tuple[int, ...], np.ndarray, tuple[int, ...]]:
        """
        Encode a constraint into token ID, reference tokens, and parameter values.

        Args:
            constraint: The constraint object.
            primitive_tokenpos: A dictionary mapping Entity ID (str) -> Start Token Index (int).

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
    def to_freecad(self, constraint: datalib.Constraint, primitives: dict[str, datalib.Entity], primitive_geoID: dict[str, tuple[int, int]]) -> list[tuple[str, list, float | None]]:
        """
        Convert a SketchGraphs Constraint to FreeCAD constraint argument tuples.

        Args:
            constraint: The SketchGraphs Constraint object.
            primitives: The SketchGraphs Entity object sets.
            primitive_geoID: Dictionary mapping reference string (e.g. "0", "0.start", "0.center")
                             -> (GeoID, PointID) tuple.

        Returns:
            List of (fc_constraint_type, args_list) tuples.
            Each tuple is used as: Sketcher.Constraint(fc_constraint_type, *args_list)
        """
        pass

    @abstractmethod
    def from_freecad(self, fc_constraint: 'Sketcher.Constraint', fc_primitives: list, geoID_to_primitive: dict[tuple[int, int], str]) -> datalib.Constraint | None:
        """
        Convert a FreeCAD constraint to a SketchGraphs Constraint.

        Args:
            fc_constraint: The FreeCAD Constraint object.
            fc_primitives: List of FreeCAD Geometry objects.
            geoID_to_primitive: Dictionary mapping (GeoID, PointID) tuple -> reference string
                                (e.g. "0", "0.start", "0.center").
        """
        pass

    def get_constraint_props(self, constraint: datalib.Constraint, expr_watermark: str = "VALUE_ASSIGNED:") -> dict[str, Any]:
        """Helper to extract a value and other properties from constraint parameters."""
        props: dict[str, Any] = {"value": None, "direction": "", "aligned": True}
        for param in constraint.parameters:
            if param.type == ConstraintParameterType.Quantity:
                if isinstance(param, QuantityParameter):
                    # NOTE: SketchGraphs data stores values not in param.value but in param.expression
                    #       Since expression is a text with a unit, it's risky to extract metric value from it.
                    #       Therefore, we assume that values in the sketch are filled by src.sketch_handler.sketch_utils.complete_constraint_values().
                    if not param.expression.startswith(expr_watermark):
                        raise ValueError(f"Value not assigned to constraint {constraint=}. Run `src.sketch_handler.sketch_utils.complete_constraint_values()` first.")
                    try:
                        props["value"] = float(param.value)
                    except:
                        continue
            # direction
            elif isinstance(param, EnumParameter) and param.parameterId == 'direction':
                props["direction"] = param.value
            # aligned
            elif isinstance(param, BooleanParameter) and param.parameterId == 'aligned':
                props["aligned"] = param.value

        if props["value"] is None:
            raise ValueError(f"No dimension constraints found: {constraint=}.")
        return props

    def create_constraint(
        self,
        constraint_type_name: str,
        refs: tuple[str, ...],
        value: float | None = None,
        direction: str | None = None,
        aligned: bool | None = None,
    ) -> datalib.Constraint:
        """Helper to create a sketchgraphs.data.Constraint object."""

        # Map to valid SketchGraphs enum; keep original name as identifier
        ctype = getattr(ConstraintType, constraint_type_name)
        params = []
        for i, ref in enumerate(refs):
            params.append(LocalReferenceParameter(parameterId=f"local{i}", value=ref))

        if value is not None:
            pid = 'angle' if constraint_type_name == 'Angle' else 'length'
            params.append(QuantityParameter(parameterId=pid, value=value, expression=str(value)))
        if direction is not None:
            assert direction in ('MINIMUM', 'HORIZONTAL', 'VERTICAL')
            params.append(EnumParameter(parameterId='direction', value=direction))
        if aligned is not None:
            params.append(BooleanParameter(parameterId='aligned', value=aligned))

        # NOTE: identifier=constraint_type_name is tentative. It's expected to be overwritten to represent a real UID.
        return Constraint(identifier=constraint_type_name, constraint_type=ctype, parameters=params)

    def _parse_references(self, constraint: Any, primitive_tokenpos: dict[str, int]) -> list[tuple[str, str | None]]:
        """
        Parse and filter references from a constraint.
        Returns a list of (entity_id, subnode_name) tuples.
        """
        refs = []
        for param in constraint.parameters:
            if not isinstance(param, LocalReferenceParameter): continue

            value = getattr(param, 'value', None)
            if not isinstance(value, str): continue

            if '.' in value:
                eid, sub = value.split('.', 1)
                if sub not in ('start', 'end', 'center'): continue
                refs.append((eid, sub))
            else:
                refs.append((value, None))

        # Filter external or truncated refs
        filtered_refs = []
        for eid, sub in refs:
            key = f"{eid}.{sub}" if sub else eid
            tokenpos = primitive_tokenpos.get(key)
            if tokenpos and tokenpos >= 0:
                filtered_refs.append((eid, sub))
            else:
                # If a referred primitive is truncated, then tokenpos = -1.
                assert tokenpos is None or tokenpos == -1, f"{tokenpos=}"
                if tokenpos is None:
                    msg = f"[ConstraintHandler._parse_reference] WARNING: didn't find {key=} in {primitive_tokenpos=}."
                    msg += "Does the key refer to an external object? Worth debugging. This is unlikely to happen."
                    print(msg, flush=True)
                filtered_refs = []
                break

        return filtered_refs

    def _get_sorted_ref_tokens(self, constraint: Any, primitive_tokenpos: dict[str, int]) -> tuple[int, ...]:
        """
        Helper to parse, sort, and convert references to tokens.
        """
        refs = self._parse_references(constraint, primitive_tokenpos)
        ref_tokens = tuple(primitive_tokenpos[f"{eid}.{sub}" if sub else eid] for eid, sub in refs)
        return self._sort_ref_tokens(ref_tokens)

    def _sort_ref_tokens(self, ref_tokens: tuple[int, ...]) -> tuple[int, ...]:
        """
        Sort reference tokens in ascending order.
        """
        return tuple(sorted(ref_tokens))

    def _get_ref_strings(self, constraint: datalib.Constraint) -> tuple[str, ...]:
        """
        Extract reference strings from constraint parameters.
        Returns list of reference strings like ["0.start", "1", "2.center"].
        """
        refs = []
        for param in constraint.parameters:
            if isinstance(param, LocalReferenceParameter):
                value = getattr(param, 'value', None)
                refs.append(value)
        return tuple(refs)

    def validate_refs(self, refs: tuple[int | str, ...]) -> bool:
        """Validate if the number of references is correct for this constraint."""
        return True # Default lenient
