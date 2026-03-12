from typing import Any, Optional, Union
import pathlib
import re
import warnings
import netCDF4
import numpy as np
from .conventions import CONVENTIONS


def _formatwarning(message, category, filename, lineno, line=None):
    return f"{category.__name__}: {message}\n"


warnings.formatwarning = _formatwarning


HDF5_MIN_SAFE_VERSION = "1.14.4"
SUSPICIOUS_EXTENSIONS = (
    ".pdf",
    ".exe",
    ".dll",
    ".so",
    ".dylib",
    ".bat",
    ".cmd",
    ".ps1",
    ".sh",
    ".py",
    ".ipynb",
    ".jar",
    ".html",
    ".js",
)
URL_PATTERN = re.compile(r"\b(?:https?|ftp|file|s3)://\S+", re.IGNORECASE)
BARE_DOMAIN_PATTERN = re.compile(
    r"\b(?:[a-z0-9-]+\.)+(?:com|net|org|edu|gov|mil|io|co|info|biz|ai|app|dev|tech|xyz)\b",
    re.IGNORECASE,
)
SUSPICIOUS_FILE_PATTERN = re.compile(
    r"(?i)(?:^|[^a-z0-9_\-\.])([a-z0-9_\-\.]+\.(?:pdf|exe|dll|so|dylib|bat|cmd|ps1|sh|py|ipynb|jar|html|js))(?=$|[^a-z0-9_\-\.])"
)


def check_sofa_against_conventions(
    target: Union[str, netCDF4.Dataset],
    convention_name: Optional[str] = None,
    version: Optional[str] = None,
):
    """Check a SOFA file against SOFA conventions.
    Emits warnings for missing mandatory fields or read-only mismatches.
    """
    dataset, _closer = _resolve_dataset(target)
    try:
        if convention_name is None:
            convention_name = getattr(dataset, "SOFAConventions", None)

        if not convention_name:
            warnings.warn("Missing SOFAConventions on dataset", UserWarning)
            return {"convention": {"name": convention_name, "version": version}}
        if convention_name not in CONVENTIONS:
            warnings.warn(
                (
                    f"Unsupported SOFAConventions '{convention_name}'. "
                    "API may not behave as expected. "
                    f"Supported: {', '.join(sorted(CONVENTIONS.keys()))}"
                ),
                UserWarning,
            )
            return {"convention": {"name": convention_name, "version": version}}

        if version is None:
            version = getattr(dataset, "SOFAConventionsVersion", None)

        if not version or version not in CONVENTIONS[convention_name]:
            warnings.warn(
                f"Unsupported or missing SOFAConventionsVersion '{version}' for {convention_name}. "
                f"Supported: {', '.join(sorted(CONVENTIONS[convention_name].keys()))}",
                UserWarning,
            )
            return {"convention": {"name": convention_name, "version": version}}

        spec = CONVENTIONS[convention_name][version]
        spec_global_attrs = {
            name.split("GLOBAL:", 1)[1] for name in spec.keys() if name.startswith("GLOBAL:")
        }
        spec_var_attrs = {
            name for name in spec.keys() if ":" in name and not name.startswith("GLOBAL:")
        }
        spec_vars = {
            name for name in spec.keys() if not name.startswith("GLOBAL:") and ":" not in name
        }
        spec_dim_letters = set()
        for entry in spec.values():
            dim_spec = entry.get("dimensions")
            if not dim_spec:
                continue
            for option in _split_dim_options(dim_spec):
                for letter in option:
                    if letter.strip():
                        spec_dim_letters.add(letter.upper())
        default_dim_letters = {"R", "E", "M", "N", "C", "I", "S"}
        expected_dim_letters = spec_dim_letters.union(default_dim_letters)

        for name, entry in spec.items():
            flags = set(entry.get("flags") or "")
            default = entry.get("default")

            kind = _parse_name(name)

            if kind[0] == "global_attr":
                attr_name = kind[1]
                exists = attr_name in dataset.ncattrs()
                if "m" in flags and not exists:
                    warnings.warn(f"Missing global attribute: {attr_name}", UserWarning)
                    continue
                if exists and "r" in flags and default not in ("", None):
                    value = getattr(dataset, attr_name)
                    if not _compare_default(default, value):
                        warnings.warn(
                            f"Global attribute {attr_name} should be '{default}', got '{value}'",
                            UserWarning,
                        )

            elif kind[0] == "var_attr":
                var_name, attr_name = kind[1], kind[2]
                if var_name not in dataset.variables:
                    if "m" in flags:
                        warnings.warn(
                            f"Missing variable for attribute: {var_name}", UserWarning
                        )
                    continue
                var = dataset.variables[var_name]
                exists = attr_name in var.ncattrs()
                if "m" in flags and not exists:
                    warnings.warn(
                        f"Missing attribute {attr_name} on variable {var_name}",
                        UserWarning,
                    )
                    continue
                if exists and "r" in flags and default not in ("", None):
                    value = getattr(var, attr_name)
                    if not _compare_default(default, value):
                        warnings.warn(
                            f"Attribute {var_name}:{attr_name} should be '{default}', got '{value}'",
                            UserWarning,
                        )

            else:
                var_name = kind[1]
                if var_name not in dataset.variables:
                    if "m" in flags:
                        warnings.warn(f"Missing variable: {var_name}", UserWarning)
                    continue
                var = dataset.variables[var_name]
                dim_spec = entry.get("dimensions")
                _warn_dim_mismatch(dataset, var_name, var, dim_spec)

                if "r" in flags and default not in ("", None):
                    value = np.array(var[:])
                    if not _compare_default(default, value):
                        warnings.warn(
                            f"Variable {var_name} does not match default value",
                            UserWarning,
                        )

        extra_global_attrs = sorted(
            attr for attr in dataset.ncattrs() if attr not in spec_global_attrs
        )
        if extra_global_attrs:
            warnings.warn(
                f"Custom global attributes found: {extra_global_attrs}",
                UserWarning,
            )

        extra_vars = sorted(
            var_name for var_name in dataset.variables.keys() if var_name not in spec_vars
        )
        if extra_vars:
            warnings.warn(
                f"Custom variables found: {extra_vars}",
                UserWarning,
            )

        extra_var_attrs: list[str] = []
        for var_name, var in dataset.variables.items():
            for attr_name in var.ncattrs():
                full_name = f"{var_name}:{attr_name}"
                if full_name not in spec_var_attrs:
                    extra_var_attrs.append(full_name)
        if extra_var_attrs:
            warnings.warn(
                f"Custom variable attributes found: {sorted(extra_var_attrs)}",
                UserWarning,
            )

        extra_dims = sorted(
            dim_name
            for dim_name in dataset.dimensions.keys()
            if dim_name.upper() not in expected_dim_letters and dim_name.upper() != "S"
        )
        if extra_dims:
            warnings.warn(
                f"Custom dimensions found: {extra_dims}",
                UserWarning,
            )

        missing_dims = sorted(
            dim for dim in expected_dim_letters if dim not in {d.upper() for d in dataset.dimensions.keys()}
        )
        if missing_dims:
            warnings.warn(
                f"Missing dimensions found: {missing_dims}",
                UserWarning,
            )

        return {"convention": {"name": convention_name, "version": version}}
    finally:
        if _closer is not None:
            _closer.close()


def check_sofa_security(
    target: Optional[Union[str, pathlib.Path, netCDF4.Dataset]] = None,
    hdf5_version: Optional[str] = None,
    min_safe_hdf5: str = HDF5_MIN_SAFE_VERSION,
    print_report: bool = True,
    paranoid_mode: bool = False,
) -> dict[str, Any]:
    """Verify SOFA/HDF5 security posture for a target file and environment."""
    report: dict[str, Any] = {
        "passed": True,
        "hdf5_version": None,
        "netcdf4_version": getattr(netCDF4, "__version__", None),
        "min_safe_hdf5": min_safe_hdf5,
        "checks": [],
        "failed": [],
    }

    def _add_check(name: str, passed: bool, message: str) -> None:
        report["checks"].append({"name": name, "passed": passed, "message": message})
        if not passed:
            report["passed"] = False
            report["failed"].append(name)

    dataset = None
    closer = None
    if not paranoid_mode and target is not None:
        try:
            dataset, closer = _resolve_dataset(target)
        except Exception as exc:
            _add_check(
                "attribute_scan",
                False,
                f"Unable to open dataset for attribute scan: {exc}",
            )
            dataset = None

    if hdf5_version is None:
        hdf5_version = _detect_hdf5_version()

    report["hdf5_version"] = hdf5_version
    if hdf5_version is None:
        _add_check(
            "hdf5_version_detected",
            False,
            "Unable to detect HDF5 library version; cannot assess CVE exposure.",
        )
        return report

    _add_check("hdf5_version_detected", True, f"HDF5 version detected: {hdf5_version}")

    version_ok = _version_ge(hdf5_version, min_safe_hdf5)
    if version_ok is None:
        _add_check(
            "hdf5_version_parse",
            False,
            f"Could not parse HDF5 version '{hdf5_version}'.",
        )
        return report

    _add_check(
        "hdf5_min_safe_version",
        version_ok,
        f"Minimum safe HDF5 version is {min_safe_hdf5}.",
    )

    _add_check(
        "risk_memory_corruption_rce",
        bool(version_ok),
        "Relies on HDF5 version meeting minimum safety baseline.",
    )
    _add_check(
        "risk_denial_of_service",
        bool(version_ok),
        "Relies on HDF5 version meeting minimum safety baseline.",
    )

    if paranoid_mode:
        if target is not None and not isinstance(target, (str, pathlib.Path)):
            raise ValueError("paranoid_mode requires a file path (str or pathlib.Path)")
        path = _path_from_target(target)
        if path is None:
            _add_check(
                "content_scan",
                False,
                "No file path provided for safe content scan.",
            )
        else:
            try:
                content = path.read_bytes()
                text = content.decode(errors="ignore")
                url_hits = _find_url_hits_in_text(text)
                if url_hits:
                    _add_check(
                        "risk_external_links_in_attributes",
                        False,
                        f"External links detected in content: {url_hits}",
                    )
                else:
                    _add_check(
                        "risk_external_links_in_attributes",
                        True,
                        "No external links detected in content.",
                    )

                extension_hits = _find_extension_hits_in_text(text)
                if extension_hits:
                    _add_check(
                        "risk_suspicious_attribute_extensions",
                        False,
                        f"Suspicious extensions detected in content: {extension_hits}",
                    )
                else:
                    _add_check(
                        "risk_suspicious_attribute_extensions",
                        True,
                        "No suspicious extensions detected in content.",
                    )
            except Exception as exc:
                _add_check(
                    "content_scan",
                    False,
                    f"Unable to scan file content safely: {exc}",
                )
    else:
        if dataset is None:
            _add_check(
                "attribute_scan",
                False,
                "No dataset available for attribute scan.",
            )
        else:
            attribute_values = _collect_attribute_strings(dataset)
            url_hits = _find_url_hits(attribute_values)
            if url_hits:
                _add_check(
                    "risk_external_links_in_attributes",
                    False,
                    f"External links detected in attributes: {url_hits}",
                )
            else:
                _add_check(
                    "risk_external_links_in_attributes",
                    True,
                    "No external links detected in attributes.",
                )

            extension_hits = _find_extension_hits(
                attribute_values, SUSPICIOUS_EXTENSIONS
            )
            if extension_hits:
                _add_check(
                    "risk_suspicious_attribute_extensions",
                    False,
                    f"Suspicious extensions detected in attributes: {extension_hits}",
                )
            else:
                _add_check(
                    "risk_suspicious_attribute_extensions",
                    True,
                    "No suspicious extensions detected in attributes.",
                )

    if closer is not None:
        closer.close()

    if paranoid_mode:
        if print_report:
            _print_security_report(report, mode="PARANOID")
        if not report["passed"]:
            raise ValueError("SOFA security check failed in paranoid mode.")
        return report

    if print_report:
        _print_security_report(report, mode="STANDARD")

    return report

def _resolve_dataset(target: Union[str, netCDF4.Dataset]):
    if hasattr(target, "netCDF4_dataset"):
        return target.netCDF4_dataset, None
    if hasattr(target, "variables") and hasattr(target, "ncattrs"):
        return target, None
    ds = netCDF4.Dataset(str(target), "r")
    return ds, ds


def _parse_name(name: str):
    if name.startswith("GLOBAL:"):
        return ("global_attr", name.split("GLOBAL:", 1)[1])
    if ":" in name:
        var, attr = name.split(":", 1)
        return ("var_attr", var, attr)
    return ("variable", name)


def _compare_default(default: Any, value: Any) -> bool:
    if default in ("", None):
        return True
    try:
        if isinstance(value, np.ndarray) or isinstance(default, (list, tuple)):
            return np.array_equal(np.array(value), np.array(default))
        return value == default
    except Exception:
        return False


def _parse_version(version: str) -> Optional[tuple[int, ...]]:
    parts = re.findall(r"\d+", version)
    if not parts:
        return None
    return tuple(int(part) for part in parts)


def _version_ge(version: str, minimum: str) -> Optional[bool]:
    parsed_version = _parse_version(version)
    parsed_minimum = _parse_version(minimum)
    if parsed_version is None or parsed_minimum is None:
        return None
    return parsed_version >= parsed_minimum


def _detect_hdf5_version() -> Optional[str]:
    candidates = [
        ("__hdf5libversion__", netCDF4),
        ("hdf5libversion", netCDF4),
    ]
    if hasattr(netCDF4, "_netCDF4"):
        candidates.extend(
            [
                ("__hdf5libversion__", netCDF4._netCDF4),
                ("hdf5libversion", netCDF4._netCDF4),
            ]
        )
    for attr_name, module in candidates:
        value = getattr(module, attr_name, None)
        if value is None:
            continue
        if callable(value):
            value = value()
        if isinstance(value, tuple):
            return ".".join(str(item) for item in value)
        if isinstance(value, bytes):
            return value.decode()
        return str(value)
    return None


def _collect_attribute_strings(dataset: netCDF4.Dataset) -> list[tuple[str, str]]:
    values: list[tuple[str, str]] = []
    for attr_name in dataset.ncattrs():
        value = getattr(dataset, attr_name)
        values.append(("GLOBAL_ATTR_NAME", attr_name))
        values.extend(_normalize_attribute_value(f"GLOBAL:{attr_name}", value))
    for var_name, var in dataset.variables.items():
        for attr_name in var.ncattrs():
            value = getattr(var, attr_name)
            values.append(("VAR_ATTR_NAME", f"{var_name}:{attr_name}"))
            values.extend(
                _normalize_attribute_value(f"{var_name}:{attr_name}", value)
            )
    return values


def _normalize_attribute_value(label: str, value: Any) -> list[tuple[str, str]]:
    normalized: list[tuple[str, str]] = []
    if isinstance(value, bytes):
        normalized.append((label, value.decode(errors="ignore")))
        return normalized
    if isinstance(value, str):
        normalized.append((label, value))
        return normalized
    if isinstance(value, (list, tuple)):
        for item in value:
            normalized.extend(_normalize_attribute_value(label, item))
        return normalized
    if isinstance(value, np.ndarray):
        for item in value.ravel().tolist():
            normalized.extend(_normalize_attribute_value(label, item))
        return normalized
    return normalized


def _find_url_hits(values: list[tuple[str, str]]) -> list[str]:
    hits: list[str] = []
    for label, text in values:
        if URL_PATTERN.search(text) or BARE_DOMAIN_PATTERN.search(text):
            hits.append(f"{label}={text}")
    return hits


def _find_extension_hits(
    values: list[tuple[str, str]],
    extensions: tuple[str, ...],
) -> list[str]:
    hits: list[str] = []
    for label, text in values:
        lower = text.lower()
        for ext in extensions:
            if ext in lower:
                hits.append(f"{label}={text}")
                break
    return hits


def _find_url_hits_in_text(text: str) -> list[str]:
    hits: list[str] = []
    for chunk in _extract_printable_runs(text):
        for match in URL_PATTERN.findall(chunk):
            hits.append(_clean_match(match))
        for match in BARE_DOMAIN_PATTERN.findall(chunk):
            cleaned = _clean_match(match)
            if cleaned and len(cleaned) >= 7:
                hits.append(cleaned)
    cleaned = [item for item in hits if item]
    return sorted(set(cleaned))


def _find_extension_hits_in_text(text: str) -> list[str]:
    hits: list[str] = []
    for chunk in _extract_printable_runs(text):
        for match in SUSPICIOUS_FILE_PATTERN.findall(chunk):
            cleaned = _clean_match(match)
            if cleaned:
                hits.append(cleaned)
    return sorted(set(hits))


def _clean_match(value: str) -> str:
    cleaned = value.split("\x00", 1)[0]
    cleaned = "".join(ch for ch in cleaned if 32 <= ord(ch) <= 126)
    return cleaned.strip()


def _extract_printable_runs(text: str, min_len: int = 6) -> list[str]:
    pattern = rf"[\x20-\x7E]{{{min_len},}}"
    return re.findall(pattern, text)


def _path_from_target(
    target: Optional[Union[str, pathlib.Path, netCDF4.Dataset]]
) -> Optional[pathlib.Path]:
    if target is None:
        return None
    if isinstance(target, (str, pathlib.Path)):
        return pathlib.Path(target)
    if hasattr(target, "filepath"):
        try:
            path_value = target.filepath()
        except Exception:
            return None
        if path_value:
            return pathlib.Path(path_value)
    return None


def _print_security_report(report: dict[str, Any], mode: str = "STANDARD") -> None:
    status = "PASSED" if report.get("passed") else "FAILED"
    print(f"Security check [{mode}]: {status}", flush=True)
    for check in report.get("checks", []):
        check_status = "PASSED" if check.get("passed") else "FAILED"
        name = check.get("name", "unknown")
        message = check.get("message", "")
        print(f"- {name}: {check_status} | {message}", flush=True)


def _split_dim_options(dimensions: Optional[str]) -> list[str]:
    if not dimensions:
        return []
    return [opt.strip() for opt in dimensions.split(",") if opt.strip()]


def _matches_dim_option(var_dims: tuple[str, ...], option: str) -> bool:
    letters = [c for c in option if c.strip()]
    if len(var_dims) != len(letters):
        return False
    for dim_name, letter in zip(var_dims, letters):
        if dim_name.upper() != letter.upper():
            return False
    return True


def _warn_dim_mismatch(dataset: netCDF4.Dataset, var_name: str, var, dim_spec: Optional[str]) -> None:
    options = _split_dim_options(dim_spec)
    if options and not any(_matches_dim_option(var.dimensions, opt) for opt in options):
        warnings.warn(
            f"Variable {var_name} has dims {var.dimensions}, expected one of {options}",
            UserWarning,
        )
    for dim_name, size in zip(var.dimensions, var.shape):
        if dim_name in dataset.dimensions:
            if dataset.dimensions[dim_name].size != size:
                warnings.warn(
                    f"Variable {var_name} dimension {dim_name} size {size} "
                    f"does not match Dimensions {dataset.dimensions[dim_name].size}",
                    UserWarning,
                )
