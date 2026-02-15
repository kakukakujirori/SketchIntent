from __future__ import annotations
from typing import Any

class Registry:
    """
    Simple Registry to map names/ids to classes/objects.
    """
    def __init__(self, name: str):
        self._name = name
        self._module_dict: dict[str, Any] = {}
        self._token_id_dict: dict[int, Any] = {}

    def register_module(self, name: str | None = None):
        def _register(cls):
            handler_name = name or cls.__name__

            # We instantiate to check properties, but register the class
            if hasattr(cls, 'name'):
                key_name = cls().name # access property from instance
                token_id = cls().token_id
            else:
                key_name = handler_name
                token_id = None

            self._module_dict[key_name] = cls
            if token_id is not None:
                self._token_id_dict[token_id] = cls
            return cls
        return _register

    def get(self, key: str) -> Any:
        return self._module_dict.get(key)

    def get_by_id(self, tid: int) -> Any:
        return self._token_id_dict.get(tid)

    @property
    def modules(self):
        return self._module_dict

PRIMITIVE_HANDLERS = Registry('primitive_handlers')
CONSTRAINT_HANDLERS = Registry('constraint_handlers')
