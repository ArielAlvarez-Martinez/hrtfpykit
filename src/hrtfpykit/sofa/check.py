from typing import Any, Optional, Union
import pathlib
import re
import netCDF4
import numpy as np
from ..utils.warnings import SOFAConventionWarning, warn_user
from .conventions import CONVENTIONS


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
SUSPICIOUS_EXTENSION_NAMES = "|".join(
    re.escape(extension.lstrip(".")) for extension in SUSPICIOUS_EXTENSIONS
)
URL_PATTERN = re.compile(r"\b(?:https?|ftp|file|s3)://\S+", re.IGNORECASE)
BARE_DOMAIN_PATTERN = re.compile(
    r"\b(?:[a-z0-9-]+\.)+(?:com|net|org|edu|gov|mil|io|co|info|biz|ai|app|dev|tech|xyz)\b",
    re.IGNORECASE,
)
SUSPICIOUS_FILE_PATTERN = re.compile(
    rf"(?i)(?:^|[^a-z0-9_\-\.])([a-z0-9_\-\.]+\.(?:{SUSPICIOUS_EXTENSION_NAMES}))(?=$|[^a-z0-9_\-\.])"
)


def check_sofa_against_conventions(
    target: Union[str, pathlib.Path, netCDF4.Dataset],
    convention_name: Optional[str] = None,
    version: Optional[str] = None,
) -> dict[str, dict[str, str | None]]:
    """Validate a SOFA file or open netCDF4 object against a SOFA convention.

    The check emits warnings for:
    - missing mandatory attributes/variables
    - read-only defaults that do not match
    - dimension mismatches
    - custom attributes/variables/dimensions not listed in the spec

    Parameters
    ----------
    target : Union[str, netCDF4.Dataset]
        Path to a SOFA file or an open netCDF4 object containing SOFA data.
    convention_name : Optional[str], optional
        Convention name to validate against. If ``None``, uses the file's
        ``SOFAConventions`` attribute.
    version : Optional[str], optional
        Convention version to validate against. If ``None``, uses the file's
        ``SOFAConventionsVersion`` attribute.

    Returns
    -------
    dict
        Summary containing the resolved convention name and version.

    Raises
    ------
    ValueError
        If ``target`` is a :class:`~hrtfpykit.sofa.SOFA` object without a
        loaded netCDF4 handle.
    OSError
        If ``target`` is a path-like input that cannot be opened as a SOFA
        file.

    Examples
    --------
    Validate a SOFA file against the convention declared in its global
    attributes and keep the resolved convention metadata for downstream checks:

    >>> from hrtfpykit.sofa import check_sofa_against_conventions
    >>> summary = check_sofa_against_conventions(
    ...     "P0001_FreeFieldComp_44kHz.sofa"
    ... )
    >>> summary["convention"]["name"]
    'SimpleFreeFieldHRIR'
    """
    dataset, _closer = _resolve_dataset(target)
    try:
        if convention_name is None:
            convention_name = getattr(dataset, "SOFAConventions", None)

        if not convention_name:
            warn_user("Missing SOFAConventions on dataset", SOFAConventionWarning)
            return {"convention": {"name": convention_name, "version": version}}
        if convention_name not in CONVENTIONS:
            warn_user(
                (
                    f"Unsupported SOFAConventions '{convention_name}'. "
                    "API may not behave as expected. "
                    f"Supported: {', '.join(sorted(CONVENTIONS.keys()))}"
                ),
                SOFAConventionWarning,
            )
            return {"convention": {"name": convention_name, "version": version}}

        if version is None:
            version = getattr(dataset, "SOFAConventionsVersion", None)

        if not version or version not in CONVENTIONS[convention_name]:
            warn_user(
                f"Unsupported or missing SOFAConventionsVersion '{version}' for {convention_name}. "
                f"Supported: {', '.join(sorted(CONVENTIONS[convention_name].keys()))}",
                SOFAConventionWarning,
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
        expected_dim_letters = spec_dim_letters

        for name, entry in spec.items():
            flags = set(entry.get("flags") or "")
            default = entry.get("default")

            kind = _parse_name(name)

            if kind[0] == "global_attr":
                attr_name = kind[1]
                exists = attr_name in dataset.ncattrs()
                if "m" in flags and not exists:
                    warn_user(
                        f"Missing global attribute: {attr_name}",
                        SOFAConventionWarning,
                    )
                    continue
                if exists and "r" in flags and default not in ("", None) and attr_name != "Version":
                    value = getattr(dataset, attr_name)
                    if not _compare_default(default, value):
                        warn_user(
                            f"Global attribute {attr_name} should be '{default}', got '{value}'",
                            SOFAConventionWarning,
                        )

            elif kind[0] == "var_attr":
                var_name, attr_name = kind[1], kind[2]
                if var_name not in dataset.variables:
                    if "m" in flags:
                        warn_user(
                            f"Missing variable for attribute: {var_name}",
                            SOFAConventionWarning,
                        )
                    continue
                var = dataset.variables[var_name]
                exists = attr_name in var.ncattrs()
                if "m" in flags and not exists:
                    warn_user(
                        f"Missing attribute {attr_name} on variable {var_name}",
                        SOFAConventionWarning,
                    )
                    continue
                if exists and "r" in flags and default not in ("", None):
                    value = getattr(var, attr_name)
                    if not _compare_default(default, value):
                        warn_user(
                            f"Attribute {var_name}:{attr_name} should be '{default}', got '{value}'",
                            SOFAConventionWarning,
                        )

            else:
                var_name = kind[1]
                if var_name not in dataset.variables:
                    if "m" in flags:
                        warn_user(
                            f"Missing variable: {var_name}",
                            SOFAConventionWarning,
                        )
                    continue
                var = dataset.variables[var_name]
                dim_spec = entry.get("dimensions")
                _warn_dim_mismatch(dataset, var_name, var, dim_spec)

                if "r" in flags and default not in ("", None):
                    value = np.array(var[:])
                    if not _compare_default(default, value):
                        warn_user(
                            f"Variable {var_name} does not match default value",
                            SOFAConventionWarning,
                        )

        extra_global_attrs = sorted(
            attr
            for attr in dataset.ncattrs()
            if attr not in spec_global_attrs and attr != "hrtfpykit"
        )
        if extra_global_attrs:
            warn_user(
                f"Custom global attributes found: {extra_global_attrs}",
                SOFAConventionWarning,
            )

        extra_vars = sorted(
            var_name for var_name in dataset.variables.keys() if var_name not in spec_vars
        )
        if extra_vars:
            warn_user(
                f"Custom variables found: {extra_vars}",
                SOFAConventionWarning,
            )

        extra_var_attrs: list[str] = []
        for var_name, var in dataset.variables.items():
            for attr_name in var.ncattrs():
                full_name = f"{var_name}:{attr_name}"
                if full_name not in spec_var_attrs:
                    extra_var_attrs.append(full_name)
        if extra_var_attrs:
            warn_user(
                f"Custom variable attributes found: {sorted(extra_var_attrs)}",
                SOFAConventionWarning,
            )

        extra_dims = sorted(
            dim_name
            for dim_name in dataset.dimensions.keys()
            if dim_name.upper() not in expected_dim_letters and dim_name.upper() != "S"
        )
        if extra_dims:
            warn_user(
                f"Custom dimensions found: {extra_dims}",
                SOFAConventionWarning,
            )

        missing_dims = sorted(
            dim for dim in expected_dim_letters if dim not in {d.upper() for d in dataset.dimensions.keys()}
        )
        if missing_dims:
            warn_user(
                f"Missing dimensions found: {missing_dims}",
                SOFAConventionWarning,
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
    """Run security checks for SOFA/HDF5 file handling.

    Checks include:

    - HDF5 runtime version against a minimum safety baseline. The default
      baseline (``HDF5_MIN_SAFE_VERSION``) is set to the first release that
      addressed a large batch of HDF5 parsing CVEs. For details, consult the
      HDF Group security advisories and the NVD CVE database for HDF5 issues.
    - detection of external links/domains and suspicious file extensions

    Modes:

    - ``STANDARD``: parse SOFA attributes using netCDF4 (opens the SOFA file)
    - ``PARANOID``: scan raw SOFA file bytes only (no parsing). Requires a path.

    Parameters
    ----------
    target : Optional[Union[str, pathlib.Path, netCDF4.Dataset]], optional
        SOFA file path or open netCDF4 object. In ``paranoid_mode``, this must be
        a path because raw bytes are inspected without parsing the file.
    hdf5_version : Optional[str], optional
        HDF5 version to validate against. If ``None``, attempts to detect the
        linked HDF5 version from netCDF4.
    min_safe_hdf5 : str, optional
        Minimum acceptable HDF5 version for baseline safety checks.
    print_report : bool, optional
        Whether to print a formatted report of all checks.
    paranoid_mode : bool, optional
        If ``True``, reads raw bytes from the file path only and never parses
        the SOFA file. Raises ValueError if checks fail.

    Returns
    -------
    dict
        Security report with overall status and individual check results.

    Raises
    ------
    ValueError
        If ``paranoid_mode`` is ``True`` and ``target`` is not a file path, or if
        paranoid mode checks fail.

    Examples
    --------
    Check a SOFA file before passing it into a loading or processing
    workflow. The report contains one entry per security check, and the
    "failed" list can be used as the error message when a file does not pass:

    >>> from hrtfpykit.sofa import check_sofa_security
    >>> report = check_sofa_security(
    ...     "P0001_FreeFieldComp_44kHz.sofa",
    ...     print_report=False,
    ... )
    >>> report["passed"]
    True
    >>> report["failed"]
    []
    >>> check_names = [check["name"] for check in report["checks"]]
    >>> "hdf5_min_safe_version" in check_names
    True
    """
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
    try:
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

                extension_hits = _find_extension_hits(attribute_values)
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

        if paranoid_mode:
            if print_report:
                _print_security_report(report, mode="PARANOID")
            if not report["passed"]:
                raise ValueError("SOFA security check failed in paranoid mode.")
            return report

        if print_report:
            _print_security_report(report, mode="STANDARD")

        return report
    finally:
        if closer is not None:
            closer.close()


def _resolve_dataset(target: Union[str, pathlib.Path, netCDF4.Dataset]):
    if hasattr(target, "netCDF4_dataset"):
        dataset = target.netCDF4_dataset
        if dataset is None:
            raise ValueError("Dataset is not loaded")
        return dataset, None
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


def _find_extension_hits(values: list[tuple[str, str]]) -> list[str]:
    hits: list[str] = []
    for label, text in values:
        if SUSPICIOUS_FILE_PATTERN.search(text):
            hits.append(f"{label}={text}")
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
    cleaned_hits = [item for item in hits if item]
    return sorted(set(cleaned_hits))


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
        warn_user(
            f"Variable {var_name} has dims {var.dimensions}, expected one of {options}",
            SOFAConventionWarning,
        )
    for dim_name, size in zip(var.dimensions, var.shape):
        if dim_name in dataset.dimensions:
            if dataset.dimensions[dim_name].size != size:
                warn_user(
                    f"Variable {var_name} dimension {dim_name} size {size} "
                    f"does not match Dimensions {dataset.dimensions[dim_name].size}",
                    SOFAConventionWarning,
                )
