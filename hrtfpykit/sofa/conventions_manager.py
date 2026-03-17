
import json
from pathlib import Path
from typing import Any, Mapping
from .conventions import CONVENTIONS

# TODO: Decide whether this should be a separate feature of hrtfpykit or if SOFA (and later HRTF) should inherit from it.

class ConventionsManager:
    """CRUD and import/export for SOFA convention specifications."""

    Spec = dict[str, dict[str, Any]]

    @staticmethod
    def _format_table(rows: list[tuple[str, str]]) -> str:
        label_width = max(len(label) for label, _ in rows)
        value_width = max(len(value) for _, value in rows)
        separator = f"-{'-' * (label_width + 2)}-{'-' * (value_width + 2)}-"
        lines = [separator]
        for label, value in rows:
            lines.append(f"| {label.ljust(label_width)} | {value.ljust(value_width)} |")
            lines.append(separator)
        return "\n".join(lines)
    
    @staticmethod
    def available_conventions_specifications() -> None:
        """Print the list of available SOFA conventions and versions.

        The table is built from the local ``CONVENTIONS`` registry.

        Raises
        ------
        ValueError
            If no conventions are registered.

        """
        if len(CONVENTIONS) is False:
            raise ValueError("There is no conventions available yet")

        rows = [("AVAILABLE CONVENTIONS", "VERSION")]
        for convention, versions in sorted(CONVENTIONS.items()):
            version_list = ", ".join(sorted(versions.keys()))
            rows.append((convention, version_list))
        table = ConventionsManager._format_table(rows)
        print(table)


    @staticmethod
    def list_conventions_specifications() -> dict[str, list[str]]:
        """List available conventions and their versions.

        Returns
        -------
        dict[str, list[str]]
            Mapping of convention name to sorted version strings.
        """
        return {
            name: sorted(versions.keys())
            for name, versions in sorted(CONVENTIONS.items())
        }

    @staticmethod
    def inspect_sofa_specification(
        name: str, version: str
    ) -> "ConventionsManager.Spec":
        """Return a convention specification.

        Parameters
        ----------
        name : str
            Convention name.
        version : str
            Convention version.

        Returns
        -------
        ConventionsManager.Spec
            Convention specification dictionary.
        """
        try:
            spec = CONVENTIONS[name][version]
        except KeyError as exc:
            raise KeyError(f"Convention '{name}' version '{version}' not found") from exc
        return spec

    @staticmethod
    def add_convention_specification(
        name: str,
        version: str,
        spec: Mapping[str, Mapping[str, Any]],
        *,
        overwrite: bool = False,
    ) -> None:
        """Add or update a convention specification.

        Parameters
        ----------
        name : str
            Convention name.
        version : str
            Convention version.
        spec : Mapping[str, Mapping[str, Any]]
            Convention specification to store.
        overwrite : bool, optional
            If True, allow overwriting an existing version.
        """
        spec_dict = ConventionsManager._validate_spec(spec)
        if name not in CONVENTIONS:
            CONVENTIONS[name] = {}
        if not overwrite and version in CONVENTIONS[name]:
            raise ValueError(f"Convention '{name}' version '{version}' already exists")
        CONVENTIONS[name][version] = spec_dict

    @staticmethod
    def delete_convention_specification_version(name: str, version: str) -> None:
        """Delete a specific version of a convention.

        Parameters
        ----------
        name : str
            Convention name.
        version : str
            Convention version to remove.
        """
        if name not in CONVENTIONS or version not in CONVENTIONS[name]:
            raise KeyError(f"Convention '{name}' version '{version}' not found")
        del CONVENTIONS[name][version]
        if not CONVENTIONS[name]:
            del CONVENTIONS[name]

    @staticmethod
    def delete_convention_specification(name: str) -> None:
        """Delete a convention and all of its versions.

        Parameters
        ----------
        name : str
            Convention name.
        """
        if name not in CONVENTIONS:
            raise KeyError(f"Convention '{name}' not found")
        del CONVENTIONS[name]

    @staticmethod
    def export_convention_specification_json(
        name: str, version: str, path: str | Path
    ) -> None:
        """Export a convention specification to JSON.

        Parameters
        ----------
        name : str
            Convention name.
        version : str
            Convention version.
        path : str | Path
            Destination file path.
        """
        spec = ConventionsManager.inspect_sofa_specification(name, version)
        payload = {"convention": name, "version": version, "spec": spec}
        out_path = Path(path)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @staticmethod
    def add_convention_specification_from_json(
        path: str | Path, *, overwrite: bool = False
    ) -> None:
        """Register conventions from a JSON file.

        The JSON can be either:
        - {"convention": "...", "version": "...", "spec": {...}}
        - {"registry": { "Name": { "1.0": {...}, ... } } }

        Parameters
        ----------
        path : str | Path
            Path to the JSON file.
        overwrite : bool, optional
            If True, allow overwriting existing versions.
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
            ConventionsManager.add_convention_specification(
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
                    ConventionsManager.add_convention_specification(
                        str(name), str(version), spec, overwrite=overwrite
                    )
            return

        raise ValueError("JSON must include a convention spec or a registry payload")

    @staticmethod
    def _validate_spec(
        spec: Mapping[str, Mapping[str, Any]],
    ) -> "ConventionsManager.Spec":
        """Validate and return a spec as a plain dict.

        Parameters
        ----------
        spec : Mapping[str, Mapping[str, Any]]
            Convention specification to validate.

        Returns
        -------
        ConventionsManager.Spec
            Normalized specification dictionary.
        """

        _required_fields = {"default", "flags", "dimensions", "type", "comment"}
        if not isinstance(spec, Mapping):
            raise ValueError("Spec must be a mapping of metadata entries")

        normalized: "ConventionsManager.Spec" = {}
        for entry_name, entry in spec.items():
            if not isinstance(entry, Mapping):
                raise ValueError(f"Entry '{entry_name}' must be a mapping")
            missing = _required_fields - set(entry.keys())
            if missing:
                missing_list = ", ".join(sorted(missing))
                raise ValueError(f"Entry '{entry_name}' is missing fields: {missing_list}")
            normalized[str(entry_name)] = dict(entry)
        return normalized
