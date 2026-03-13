"""Manage SOFA conventions specifications.

This module provides a small registry class to add, remove, inspect, and
serialize SOFA convention specifications without editing ``conventions.py``.

"""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from .conventions import CONVENTIONS


class ConventionsManager:
    """CRUD and import/export for SOFA convention specifications."""

    Spec = dict[str, dict[str, Any]]
    Registry = dict[str, dict[str, "ConventionsManager.Spec"]]

    def __init__(
        self,
        registry: "ConventionsManager.Registry" | None = None,
    ) -> None:
        """Initialize the manager.

        Parameters
        ----------
        registry : Registry | None
            Optional registry to seed the manager with.
        copy_on_get : bool
            If True, ``get_convention`` returns a deep copy.
        """
        self._registry: "ConventionsManager.Registry" = (
            deepcopy(registry) if registry is not None else deepcopy(CONVENTIONS)
        )

    def list_conventions_specifications(self) -> dict[str, list[str]]:
        """List available conventions and their versions."""
        return {
            name: sorted(versions.keys())
            for name, versions in sorted(self._registry.items())
        }

    def inspect_sofa_specification(
        self, name: str, version: str
    ) -> "ConventionsManager.Spec":
        """Return a convention specification."""
        try:
            spec = self._registry[name][version]
        except KeyError as exc:
            raise KeyError(f"Convention '{name}' version '{version}' not found") from exc
        return deepcopy(spec)

    def add_convention_specification(
        self,
        name: str,
        version: str,
        spec: Mapping[str, Mapping[str, Any]],
        *,
        overwrite: bool = False,
    ) -> None:
        """Add or update a convention specification."""
        spec_dict = self._validate_spec(spec)
        if name not in self._registry:
            self._registry[name] = {}
        if not overwrite and version in self._registry[name]:
            raise ValueError(f"Convention '{name}' version '{version}' already exists")
        self._registry[name][version] = deepcopy(spec_dict)

    def delete_convention_specification_version(self, name: str, version: str) -> None:
        """Delete a specific version of a convention."""
        if name not in self._registry or version not in self._registry[name]:
            raise KeyError(f"Convention '{name}' version '{version}' not found")
        del self._registry[name][version]
        if not self._registry[name]:
            del self._registry[name]

    def delete_convention_specification(self, name: str) -> None:
        """Delete a convention and all of its versions."""
        if name not in self._registry:
            raise KeyError(f"Convention '{name}' not found")
        del self._registry[name]

    def export_convention_specification_json(
        self, name: str, version: str, path: str | Path
    ) -> None:
        """Export a convention specification to JSON."""
        spec = self.inspect_sofa_specification(name, version)
        payload = {"convention": name, "version": version, "spec": spec}
        out_path = Path(path)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def add_convention_specification_from_json(
        self, path: str | Path, *, overwrite: bool = False
    ) -> None:
        """Register conventions from a JSON file.

        The JSON can be either:
        - {"convention": "...", "version": "...", "spec": {...}}
        - {"registry": { "Name": { "1.0": {...}, ... } } }
        """
        in_path = Path(path)
        if not in_path.exists():
            raise FileNotFoundError(f"JSON file not found: {in_path}")
        payload = json.loads(in_path.read_text(encoding="utf-8"))

        if isinstance(payload, dict) and (
            "convention" in payload or "name" in payload
        ):
            name = payload.get("convention") or payload.get("name")
            version = payload.get("version")
            spec = payload.get("spec")
            if not name or not version or spec is None:
                raise ValueError("JSON must include 'convention' (or 'name'), 'version', and 'spec'")
            self.add_convention_specification(
                str(name), str(version), spec, overwrite=overwrite
            )
            return

        if isinstance(payload, dict) and "registry" in payload:
            registry = payload["registry"]
            if not isinstance(registry, Mapping):
                raise ValueError("'registry' must be a mapping of conventions")
            for name, versions in registry.items():
                if not isinstance(versions, Mapping):
                    raise ValueError(f"Registry entry for '{name}' must be a mapping")
                for version, spec in versions.items():
                    self.add_convention_specification(
                        str(name), str(version), spec, overwrite=overwrite
                    )
            return

        raise ValueError("JSON must include a convention spec or a registry payload")

    @staticmethod
    def _validate_spec(
        spec: Mapping[str, Mapping[str, Any]],
    ) -> "ConventionsManager.Spec":
        """Validate and return a spec as a plain dict."""
        if not isinstance(spec, Mapping):
            raise ValueError("Spec must be a mapping of metadata entries")

        normalized: "ConventionsManager.Spec" = {}
        _required_fields = {"default", "flags", "dimensions", "type", "comment"}
        for entry_name, entry in spec.items():
            if not isinstance(entry, Mapping):
                raise ValueError(f"Entry '{entry_name}' must be a mapping")
            missing = ConventionsManager._required_fields - set(entry.keys())
            if missing:
                missing_list = ", ".join(sorted(missing))
                raise ValueError(f"Entry '{entry_name}' is missing fields: {missing_list}")
            normalized[str(entry_name)] = dict(entry)
        return normalized


