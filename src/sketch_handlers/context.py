from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Literal, TYPE_CHECKING
import enum
import importlib
import pkgutil

from .registry import PRIMITIVE_HANDLERS, CONSTRAINT_HANDLERS

if TYPE_CHECKING:
    from .primitives.base import PrimitiveHandler
    from .constraints.base import ConstraintHandler


@dataclass
class SketchContext:
    """
    Context object holding configuration and registry for tokenization/detokenization.
    """
    # Configuration
    max_token_length: int | None = 256
    num_quantization_bins: int = 64
    align_constraint_token_length_to: int | None = None
    use_stop_padding: bool = False
    constraint_order: Literal["sort_by_ref_id", "random_shuffle", "original"] = "sort_by_ref_id"

    # Registry maps (Instantiated Handlers)
    _primitive_handlers: dict[str, 'PrimitiveHandler'] = field(default_factory=dict, init=False)
    _constraint_handlers: dict[str, 'ConstraintHandler'] = field(default_factory=dict, init=False)

    _token_id_to_primitive_handler: dict[int, 'PrimitiveHandler'] = field(default_factory=dict, init=False)
    _token_id_to_constraint_handler: dict[int, 'ConstraintHandler'] = field(default_factory=dict, init=False)

    def register_primitive_handler(self, handler: 'PrimitiveHandler'):
        """Registers an *instance* of a handler for runtime lookup."""
        self._primitive_handlers[handler.name] = handler
        self._token_id_to_primitive_handler[handler.token_id] = handler

    def register_constraint_handler(self, handler: 'ConstraintHandler'):
        """Registers an *instance* of a handler for runtime lookup."""
        self._constraint_handlers[handler.name] = handler
        self._token_id_to_constraint_handler[handler.token_id] = handler

    def get_primitive_handler(self, name: str) -> 'PrimitiveHandler' | None:
        return self._primitive_handlers.get(name)

    def get_constraint_handler(self, name: str) -> 'ConstraintHandler' | None:
        return self._constraint_handlers.get(name)

    def __post_init__(self):
        # Import primitives
        import src.sketch_handlers.primitives as primitives_pkg
        for _, name, _ in pkgutil.iter_modules(primitives_pkg.__path__):
            importlib.import_module(f'src.sketch_handlers.primitives.{name}')

        # Import constraints
        import src.sketch_handlers.constraints as constraints_pkg
        for _, name, _ in pkgutil.iter_modules(constraints_pkg.__path__):
            importlib.import_module(f'src.sketch_handlers.constraints.{name}')

        # Instantiate from Registry
        for cls in PRIMITIVE_HANDLERS.modules.values():
            self.register_primitive_handler(cls())

        for cls in CONSTRAINT_HANDLERS.modules.values():
            self.register_constraint_handler(cls())
