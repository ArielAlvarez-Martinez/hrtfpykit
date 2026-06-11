from typing import TYPE_CHECKING, Any, cast

import numpy as np
from ..utils.coordinates import (
    cartesian_to_lateral_polar,
    cartesian_to_spherical,
    get_closest_position_index,
    get_named_positions,
    get_spherical_positions,
    lateral_polar_to_cartesian,
    lateral_polar_to_spherical,
    spherical_to_cartesian,
    spherical_to_lateral_polar,
)
from ..utils.planes import get_frontal_plane, get_horizontal_plane, get_median_plane

if TYPE_CHECKING:
    from .hrtf import HRTF


class Sources:
    def __init__(
        self,
        hrtf: "HRTF | None" = None,
    ) -> None:
        """Manage source positions, coordinate systems, and spatial selections.

        :class:`~hrtfpykit.hrtf.sources.Sources` is the source-position
        manager used by :class:`~hrtfpykit.hrtf.HRTF`. It stores an
        in-memory copy of SOFA ``SourcePosition`` values and their
        ``SourcePosition:Type`` and ``SourcePosition:Units`` attributes,
        converts positions on demand, and resolves source-grid queries used
        by selection, metrics, spherical harmonics, and plotting utilities.

        The manager stores the target coordinate system in
        :attr:`~hrtfpykit.hrtf.sources.Sources.source_coordinate_system`.
        Changing that value changes how
        :meth:`~hrtfpykit.hrtf.sources.Sources.get_positions` and query
        methods expose positions; it does not rewrite the stored SOFA
        ``SourcePosition`` array by itself. Spatial subsets created
        through :meth:`~hrtfpykit.hrtf.HRTF.select` are also respected, so
        returned arrays and matched indices refer to the current HRTF view
        rather than necessarily to every source in the original SOFA file.

        Notes
        -----
        Conventions implemented by this class:

        - Spherical (SOFA-style): (azimuth, elevation, radius) with azimuth in
          [0, 360) degrees (anticlockwise in horizontal plane), elevation in
          [-90, 90] degrees (positive up), and non-negative radius.
        - Lateral-polar: (lateral, polar, radius) with lateral in [-90, 90]
          degrees (positive left), polar normalized to [-90, 270) degrees, and
          non-negative radius.
        - Cartesian: (x, y, z) with +y as left and +z as up.

        At lateral poles (abs(lateral) == 90) and at zero radius, polar is singular.
        This implementation uses a deterministic placeholder polar = 0.

        Parameters
        ----------
        hrtf : :class:`~hrtfpykit.hrtf.HRTF` | None, default=None
            Owning HRTF instance. Most user code obtains this object from
            :attr:`~hrtfpykit.hrtf.HRTF.Sources`. When provided, the HRTF
            must have a loaded :class:`~hrtfpykit.sofa.SOFA` object
            containing ``SourcePosition`` metadata.

        Attributes
        ----------
        source_coordinate_system : str
            Target coordinate system used by source-grid query methods.
        _selected_indices : numpy.ndarray or None
            Source-position indices retained by the current HRTF view after spatial
            selection.

        Examples
        --------
        Load an HRTF and inspect the source manager exposed by the HRTF object:

        >>> from hrtfpykit.hrtf import load_hrtf
        >>> hrtf = load_hrtf("P0001_FreeFieldComp_44kHz.sofa")
        >>> hrtf.Sources.source_coordinate_system
        'spherical'
        >>> hrtf.Sources.get_positions().shape
        (793, 3)
        """
        self._hrtf = hrtf
        self.positions: np.ndarray | None = None
        self.position_coordinate_system: str | None = None
        self.position_units: str | None = None
        self.source_coordinate_system: str | None = None
        self._selected_indices: np.ndarray | None = None
        if self._hrtf is not None and self._hrtf.Sofa is not None:
            if self._hrtf.Sofa.Variables is None or self._hrtf.Sofa.VariableAttributes is None:
                raise ValueError("SOFA dataset is not loaded")
            self.positions = np.asarray(
                cast(
                    Any,
                    cast(Any, self._hrtf.Sofa.Variables).get("SourcePosition"),
                ).value,
                dtype=float,
            )
            self.position_coordinate_system = str(
                cast(
                    Any,
                    cast(Any, self._hrtf.Sofa.VariableAttributes).get("SourcePosition:Type"),
                ).value
            )
            self.position_units = str(
                cast(
                    Any,
                    cast(Any, self._hrtf.Sofa.VariableAttributes).get("SourcePosition:Units"),
                ).value
            )
            self.source_coordinate_system = self.position_coordinate_system

    def get_positions(
        self,
        angle_unit: str = "degrees",
        coordinate_system: str = "spherical",
        plane: str | None = None,
        plane_angle: float | None = None,
    ) -> np.ndarray:
        """Return the current source grid in the requested coordinate system.

        Positions are read from the in-memory ``SourcePosition`` snapshot
        stored on this object. The source coordinate system is taken from
        ``SourcePosition:Type`` and converted on read. By default, positions
        are returned in spherical coordinates. If the owning HRTF has been
        spatially selected, only the selected source rows are returned. If
        ``plane`` is provided, hrtfpykit
        first resolves the nearest measured plane, keeps only those source
        rows, and then returns them in ``coordinate_system``.

        Parameters
        ----------
        angle_unit : {``degrees``, ``radians``}, default=``degrees``
            Angular unit used for returned spherical or lateral-polar angles.
            Cartesian coordinates are returned in their stored distance unit.
        coordinate_system : {``spherical``, ``cartesian``, ``lateral-polar``}, default=``spherical``
            Coordinate system used for returned positions.
        plane : {``horizontal``, ``median``, ``frontal``} or None, default=None
            Optional source plane used to filter returned positions. Plane
            matching is independent of the requested output
            ``coordinate_system``.

            - ``"horizontal"`` selects a constant spherical elevation.
            - ``"median"`` selects a constant lateral-polar lateral angle.
            - ``"frontal"`` selects the nearest requested spherical azimuth and
              the nearest opposite azimuth.
        plane_angle : float or None, default=None
            Requested plane coordinate in ``angle_unit``. For
            ``plane="horizontal"`` this is spherical elevation. For
            ``plane="median"`` this is lateral-polar lateral angle. For
            ``plane="frontal"`` this is spherical azimuth; the opposite azimuth
            is added automatically. ``None`` uses 0 for horizontal and median
            planes, and 90 degrees or pi / 2 radians for the frontal plane.

        Returns
        -------
        np.ndarray
            Source-position array with shape (N, 3). The columns are
            (azimuth, elevation, radius) for spherical,
            (lateral, polar, radius) for lateral-polar, or (x, y, z)
            for cartesian coordinates.

        Raises
        ------
        ValueError
            If angle_unit is unsupported, plane is unsupported, the source or
            target coordinate system is unsupported, SOFA angular units cannot be
            interpreted, or the requested conversion is not implemented.

        Notes
        -----
        SOFA angular units are detected from the ``SourcePosition:Units``
        attribute. Angular source data stored in radians are converted through
        cartesian coordinates when degree/radian conversion is required.

        Examples
        --------
        Load an HRTF, inspect the full spherical source grid, then request
        plane-filtered positions in spherical and Cartesian coordinates:

        >>> from hrtfpykit.hrtf import load_hrtf
        >>> hrtf = load_hrtf("P0001_FreeFieldComp_44kHz.sofa")
        >>> hrtf.Sources.get_positions().shape
        (793, 3)
        >>> horizontal = hrtf.Sources.get_positions(plane="horizontal")
        >>> horizontal.shape
        (72, 3)
        >>> frontal_xyz = hrtf.Sources.get_positions(
        ...     coordinate_system="cartesian",
        ...     plane="frontal",
        ... )
        >>> frontal_xyz.shape
        (22, 3)
        """
        if self.positions is None:
            raise ValueError("Source positions are not available")
        if self.position_coordinate_system is None:
            raise ValueError("Source position coordinate system is not available")
        if self.position_units is None:
            raise ValueError("Source position units are not available")
        source_positions = np.array(self.positions, dtype=float, copy=True)
        source_system = self.position_coordinate_system
        source_units = self.position_units
        target_system = coordinate_system

        requested_angle_unit = str(angle_unit).strip().lower()
        if requested_angle_unit not in {"degrees", "radians"}:
            raise ValueError("angle_unit must be 'degrees' or 'radians'")

        source_system = str(source_system).strip().lower()
        target_system = str(target_system).strip().lower()
        allowed_coordinate_systems = {"spherical", "cartesian", "lateral-polar"}
        if source_system not in allowed_coordinate_systems:
            raise ValueError(f"Unsupported source coordinate system: {source_system!r}")
        if target_system not in allowed_coordinate_systems:
            raise ValueError(f"Unsupported target coordinate system: {target_system!r}")

        source_units = str(source_units).strip().lower()
        source_angle_unit = "degrees"
        if "radian" in source_units:
            source_angle_unit = "radians"
        elif "degree" not in source_units:
            if source_system != "cartesian":
                raise ValueError(
                    "SourcePosition:Units must include degree or radian for angular coordinate systems"
                )

        source_positions = np.asarray(source_positions, dtype=float)
        if self._selected_indices is not None:
            source_positions = np.take(
                source_positions,
                np.asarray(self._selected_indices, dtype=int),
                axis=0,
            )

        if plane is not None:
            hrtf = self._hrtf
            if hrtf is None:
                raise ValueError("Plane selection requires an owning HRTF")
            plane_key = str(plane).strip().lower()
            if plane_key == "horizontal":
                selected_plane_angle = 0.0 if plane_angle is None else float(plane_angle)
                plane_indices, _ = get_horizontal_plane(
                    hrtf,
                    plane_angle=selected_plane_angle,
                    angle_unit=requested_angle_unit,
                )
            elif plane_key == "median":
                selected_plane_angle = 0.0 if plane_angle is None else float(plane_angle)
                plane_indices, _ = get_median_plane(
                    hrtf,
                    plane_angle=selected_plane_angle,
                    angle_unit=requested_angle_unit,
                )
            elif plane_key == "frontal":
                if plane_angle is None:
                    selected_plane_angle = 90.0 if requested_angle_unit == "degrees" else np.pi / 2.0
                else:
                    selected_plane_angle = float(plane_angle)
                plane_indices, _ = get_frontal_plane(
                    hrtf,
                    plane_angle=selected_plane_angle,
                    angle_unit=requested_angle_unit,
                )
            else:
                raise ValueError("plane must be one of: horizontal, median, frontal")
            source_positions = np.take(source_positions, plane_indices, axis=0)

        if source_system == target_system:
            if target_system == "cartesian" or source_angle_unit == requested_angle_unit:
                return np.asarray(source_positions, dtype=float)
            if target_system == "spherical":
                cartesian = spherical_to_cartesian(
                    source_positions,
                    angle_unit=source_angle_unit,
                )
                return np.asarray(
                    cartesian_to_spherical(
                        cartesian,
                        angle_unit=requested_angle_unit,
                    ),
                    dtype=float,
                )
            cartesian = lateral_polar_to_cartesian(
                source_positions,
                angle_unit=source_angle_unit,
            )
            return np.asarray(
                cartesian_to_lateral_polar(
                    cartesian,
                    angle_unit=requested_angle_unit,
                ),
                dtype=float,
            )

        if source_system == "spherical" and target_system == "cartesian":
            return np.asarray(
                spherical_to_cartesian(
                    source_positions,
                    angle_unit=source_angle_unit,
                ),
                dtype=float,
            )
        if source_system == "cartesian" and target_system == "spherical":
            return np.asarray(
                cartesian_to_spherical(
                    source_positions,
                    angle_unit=requested_angle_unit,
                ),
                dtype=float,
            )
        if source_system == "cartesian" and target_system == "lateral-polar":
            return np.asarray(
                cartesian_to_lateral_polar(
                    source_positions,
                    angle_unit=requested_angle_unit,
                ),
                dtype=float,
            )
        if source_system == "lateral-polar" and target_system == "cartesian":
            return np.asarray(
                lateral_polar_to_cartesian(
                    source_positions,
                    angle_unit=source_angle_unit,
                ),
                dtype=float,
            )
        if source_system == "spherical" and target_system == "lateral-polar":
            cartesian = spherical_to_cartesian(
                source_positions,
                angle_unit=source_angle_unit,
            )
            return np.asarray(
                cartesian_to_lateral_polar(
                    cartesian,
                    angle_unit=requested_angle_unit,
                ),
                dtype=float,
            )
        if source_system == "lateral-polar" and target_system == "spherical":
            cartesian = lateral_polar_to_cartesian(
                source_positions,
                angle_unit=source_angle_unit,
            )
            return np.asarray(
                cartesian_to_spherical(
                    cartesian,
                    angle_unit=requested_angle_unit,
                ),
                dtype=float,
            )

        raise ValueError(
            f"Unsupported conversion from {source_system!r} to {target_system!r}"
        )

    def get_azimuth_angles(
        self,
        angle_unit: str = "degrees",
    ) -> np.ndarray:
        """Return unique azimuth values available in the current source grid.

        The source grid is first normalized to spherical coordinates. This
        makes the result independent of the active source_coordinate_system
        while still respecting any spatial subset selected on the owning HRTF.

        Parameters
        ----------
        angle_unit : {``degrees``, ``radians``}, default=``degrees``
            Angular unit used for returned azimuth values.

        Returns
        -------
        np.ndarray
            One-dimensional array of sorted unique azimuth angles rounded to
            two decimals.

        Raises
        ------
        ValueError
            If positions cannot be read or converted to spherical coordinates.

        Examples
        --------
        Load an HRTF and inspect the available azimuth grid in degrees:

        >>> from hrtfpykit.hrtf import load_hrtf
        >>> hrtf = load_hrtf("P0001_FreeFieldComp_44kHz.sofa")
        >>> hrtf.Sources.get_azimuth_angles()[:5]
        array([ 0.,  5., 10., 15., 20.])
        """
        spherical = get_spherical_positions(self, angle_unit=angle_unit)
        azimuth = spherical[..., 0]
        return np.unique(np.round(np.asarray(azimuth, dtype=float), 2))

    def get_elevation_angles(
        self,
        angle_unit: str = "degrees",
    ) -> np.ndarray:
        """Return unique elevation values available in the current source grid.

        The source grid is first normalized to spherical coordinates. This
        makes the result independent of the active source_coordinate_system
        while still respecting any spatial subset selected on the owning HRTF.

        Parameters
        ----------
        angle_unit : {``degrees``, ``radians``}, default=``degrees``
            Angular unit used for returned elevation values.

        Returns
        -------
        np.ndarray
            One-dimensional array of sorted unique elevation angles rounded to
            two decimals.

        Raises
        ------
        ValueError
            If positions cannot be read or converted to spherical coordinates.

        Examples
        --------
        Load an HRTF and inspect the available elevation grid in degrees:

        >>> from hrtfpykit.hrtf import load_hrtf
        >>> hrtf = load_hrtf("P0001_FreeFieldComp_44kHz.sofa")
        >>> hrtf.Sources.get_elevation_angles()
        array([-45., -30., -20., -10.,   0.,  10.,  20.,  30.,  45.,  60.,  75.,
                90.])
        """
        spherical = get_spherical_positions(self, angle_unit=angle_unit)
        elevation = spherical[..., 1]
        return np.unique(np.round(np.asarray(elevation, dtype=float), 2))

    def get_elevation_angles_for_azimuth(
        self,
        azimuth: float,
        angle_unit: str = "degrees",
    ) -> tuple[np.ndarray, float]:
        """Return elevations available at the nearest source-grid azimuth.

        This method is useful for plane-based plotting and selection UIs where
        a requested azimuth may not exist exactly in the measured source grid.
        The azimuth match is circular, so values near 0 and 360
        degrees, or 0 and 2*pi radians, are treated as neighbors.

        Parameters
        ----------
        azimuth : float
            Requested azimuth angle used to query the source grid.
        angle_unit : {``degrees``, ``radians``}, default=``degrees``
            Angular unit for azimuth, returned elevations, and
            real_azimuth.

        Returns
        -------
        tuple[np.ndarray, float]
            (elevation_angles, real_azimuth) where elevation_angles is a
            one-dimensional array of unique elevations available at the matched
            azimuth, and real_azimuth is the actual azimuth selected from
            the grid. Both outputs are rounded to two decimals.

        Raises
        ------
        ValueError
            If azimuth is boolean or non-finite, angle_unit is
            unsupported, or positions cannot be read or converted to spherical
            coordinates.

        Examples
        --------
        Ask for the elevations available at the source-grid azimuth nearest to
        0 degrees:

        >>> from hrtfpykit.hrtf import load_hrtf
        >>> hrtf = load_hrtf("P0001_FreeFieldComp_44kHz.sofa")
        >>> elevations, real_azimuth = hrtf.Sources.get_elevation_angles_for_azimuth(0.0)
        >>> elevations[:5]
        array([-45., -30., -20., -10.,   0.])
        >>> real_azimuth
        0.0
        """
        if isinstance(azimuth, bool):
            raise ValueError("azimuth must be a finite value")
        azimuth = float(azimuth)
        if not np.isfinite(azimuth):
            raise ValueError("azimuth must be a finite value")

        spherical = get_spherical_positions(self, angle_unit=angle_unit)

        unit = str(angle_unit).strip().lower()
        if unit not in {"degrees", "radians"}:
            raise ValueError("angle_unit must be 'degrees' or 'radians'")

        azimuth_angles = np.asarray(spherical[..., 0], dtype=float)
        elevation_angles = np.asarray(spherical[..., 1], dtype=float)
        available_azimuths = np.unique(azimuth_angles)
        full = 360.0 if unit == "degrees" else 2.0 * np.pi
        half = full / 2.0
        azimuth_deltas = np.mod(available_azimuths - azimuth + half, full) - half
        real_azimuth = float(available_azimuths[int(np.argmin(np.abs(azimuth_deltas)))])
        selected = np.isclose(
            np.mod(azimuth_angles - real_azimuth + half, full) - half,
            0.0,
            atol=1e-8,
            rtol=0.0,
        )
        return np.unique(np.round(elevation_angles[selected], 2)), round(real_azimuth, 2)

    def get_azimuth_angles_for_elevation(
        self,
        elevation: float,
        angle_unit: str = "degrees",
    ) -> tuple[np.ndarray, float]:
        """Return azimuths available at the nearest source-grid elevation.

        This method is useful for horizontal-plane workflows where the
        requested elevation may not exist exactly in the measured source grid.
        Elevation matching uses the nearest available numerical elevation in
        spherical coordinates.

        Parameters
        ----------
        elevation : float
            Requested elevation angle used to query the source grid.
        angle_unit : {``degrees``, ``radians``}, default=``degrees``
            Angular unit for elevation, returned azimuths, and
            real_elevation.

        Returns
        -------
        tuple[np.ndarray, float]
            (azimuth_angles, real_elevation) where azimuth_angles is a
            one-dimensional array of unique azimuths available at the matched
            elevation, and real_elevation is the actual elevation selected
            from the grid. Both outputs are rounded to two decimals.

        Raises
        ------
        ValueError
            If elevation is boolean or non-finite, angle_unit is
            unsupported, or positions cannot be read or converted to spherical
            coordinates.

        Examples
        --------
        Ask for the azimuths available at the source-grid elevation nearest to
        the horizontal plane:

        >>> from hrtfpykit.hrtf import load_hrtf
        >>> hrtf = load_hrtf("P0001_FreeFieldComp_44kHz.sofa")
        >>> azimuths, real_elevation = hrtf.Sources.get_azimuth_angles_for_elevation(0.0)
        >>> azimuths[:5]
        array([ 0.,  5., 10., 15., 20.])
        >>> real_elevation
        0.0
        """
        if isinstance(elevation, bool):
            raise ValueError("elevation must be a finite value")
        elevation = float(elevation)
        if not np.isfinite(elevation):
            raise ValueError("elevation must be a finite value")

        spherical = get_spherical_positions(self, angle_unit=angle_unit)

        elevation_angles = np.asarray(spherical[..., 1], dtype=float)
        azimuth_angles = np.asarray(spherical[..., 0], dtype=float)
        available_elevations = np.unique(elevation_angles)
        real_elevation = float(
            available_elevations[int(np.argmin(np.abs(available_elevations - elevation)))]
        )
        selected = np.isclose(
            elevation_angles,
            real_elevation,
            atol=1e-8,
            rtol=0.0,
        )
        return np.unique(np.round(azimuth_angles[selected], 2)), round(real_elevation, 2)

    def get_position_index(
        self,
        position: np.ndarray | list[float] | tuple[float, float, float] | str,
        coordinate_system: str = "spherical",
        angle_unit: str = "degrees",
    ) -> tuple[int, np.ndarray]:
        """Return the nearest source index and its resolved grid position.

        The query is matched against the current source grid, including any
        source subset already selected on the owning :class:`~hrtfpykit.hrtf.HRTF` object. Numeric
        positions are interpreted in coordinate_system. Named positions use
        the canonical horizontal spherical aliases ``front``, ``back``,
        ``left``, and ``right`` and are then returned in the requested
        coordinate system.

        Parameters
        ----------
        position : np.ndarray | list[float] | tuple[float, float, float] | str
            Query position. Numeric spherical and lateral-polar queries may be
            angle-only (2,) or full (3,) coordinates. Cartesian queries
            must be (3,). String queries must be one of the supported named
            positions.
        coordinate_system : {``spherical``, ``cartesian``, ``lateral-polar``}, default=``spherical``
            Coordinate system of numeric position queries and returned
            real_position.
        angle_unit : {``degrees``, ``radians``}, default=``degrees``
            Angular unit for spherical/lateral-polar inputs and outputs.

        Returns
        -------
        tuple[int, np.ndarray]
            (idx, real_position) where idx is the nearest source index
            in the current source view and real_position is the matched
            grid coordinate rounded to two decimals in coordinate_system.

        Raises
        ------
        ValueError
            If the coordinate system or angle unit is unsupported, source
            positions do not form an (N, 3) grid, a named position is
            unknown, a query has an invalid shape, or the needed coordinate
            conversion is unsupported.

        Examples
        --------
        Resolve a named source direction to the nearest measured source index
        and its real spherical grid position:

        >>> from hrtfpykit.hrtf import load_hrtf
        >>> hrtf = load_hrtf("P0001_FreeFieldComp_44kHz.sofa")
        >>> hrtf.Sources.get_position_index("front")
        (4, array([0. , 0. , 1.5]))
        """
        system = str(coordinate_system).strip().lower()
        if system not in {"spherical", "cartesian", "lateral-polar"}:
            raise ValueError(
                "coordinate_system must be one of: spherical, cartesian, lateral-polar"
            )
        unit = str(angle_unit).strip().lower()
        if unit not in {"degrees", "radians"}:
            raise ValueError("angle_unit must be 'degrees' or 'radians'")

        grid_system = str(self.source_coordinate_system).strip().lower()
        grid_positions = self.get_positions(
            angle_unit=unit,
            coordinate_system=grid_system,
        )
        if grid_positions.ndim != 2 or grid_positions.shape[-1] != 3:
            raise ValueError("Source positions grid must have shape (N, 3)")

        if grid_system == system:
            grid_in_query_system = grid_positions
        elif grid_system == "cartesian" and system == "spherical":
            grid_in_query_system = cartesian_to_spherical(
                grid_positions,
                angle_unit=unit,
            )
        elif grid_system == "cartesian" and system == "lateral-polar":
            grid_in_query_system = cartesian_to_lateral_polar(
                grid_positions,
                angle_unit=unit,
            )
        elif grid_system == "spherical" and system == "cartesian":
            grid_in_query_system = spherical_to_cartesian(
                grid_positions,
                angle_unit=unit,
            )
        elif grid_system == "spherical" and system == "lateral-polar":
            grid_in_query_system = spherical_to_lateral_polar(
                grid_positions,
                angle_unit=unit,
            )
        elif grid_system == "lateral-polar" and system == "cartesian":
            grid_in_query_system = lateral_polar_to_cartesian(
                grid_positions,
                angle_unit=unit,
            )
        elif grid_system == "lateral-polar" and system == "spherical":
            grid_in_query_system = lateral_polar_to_spherical(
                grid_positions,
                angle_unit=unit,
            )
        else:
            raise ValueError(
                f"Unsupported conversion from {grid_system!r} to {system!r}"
            )

        if isinstance(position, str):
            named_position = str(position).strip().lower()
            named_positions = get_named_positions(angle_unit=unit)
            if named_position not in named_positions:
                raise ValueError(
                    "named position accepts: front, back, left, or right"
                )
            if grid_system == "spherical":
                grid_in_spherical = grid_positions
            elif grid_system == "cartesian":
                grid_in_spherical = cartesian_to_spherical(
                    grid_positions,
                    angle_unit=unit,
                )
            elif grid_system == "lateral-polar":
                grid_in_spherical = lateral_polar_to_spherical(
                    grid_positions,
                    angle_unit=unit,
                )
            else:
                raise ValueError(
                    f"Unsupported conversion from {grid_system!r} to 'spherical'"
                )
            query_position = np.asarray(named_positions[named_position], dtype=float)
            idx = get_closest_position_index(
                query_position=query_position,
                grid_positions=grid_in_spherical,
                coordinate_system="spherical",
                angle_unit=unit,
            )
            return idx, np.round(np.asarray(grid_in_query_system[idx], dtype=float), 2)

        query_position = np.asarray(position, dtype=float)
        idx = get_closest_position_index(
            query_position=query_position,
            grid_positions=grid_in_query_system,
            coordinate_system=system,
            angle_unit=unit,
        )
        return idx, np.round(np.asarray(grid_in_query_system[idx], dtype=float), 2)
