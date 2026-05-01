from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np


class HRTFSpec:
    def __init__(
        self,
        domain: str = "time",
        signal: str = "ir",
        positions: str | tuple[int, ...] | list[int] | np.ndarray = "all",
        plane: str | tuple[object, ...] | dict[str, object] | None = None,
        ears: str | tuple[str, ...] = "both",
        index_by: str | tuple[str, ...] = ("subject",),
        position_one_hot: bool = False,
        position_index: bool = False,
        ear_one_hot: bool = False,
        ear_index: bool = False,
        frequency_one_hot: bool = False,
        frequency_index: bool = False,
        sample_one_hot: bool = False,
        sample_index: bool = False,
        transform: Callable | None = None,
        name: str | None = None,
    ) -> None:
        self.domain = domain
        self.signal = signal
        self.positions = positions
        self.plane = plane
        self.ears = ears
        self.index_by = index_by
        self.position_one_hot = position_one_hot
        self.position_index = position_index
        self.ear_one_hot = ear_one_hot
        self.ear_index = ear_index
        self.frequency_one_hot = frequency_one_hot
        self.frequency_index = frequency_index
        self.sample_one_hot = sample_one_hot
        self.sample_index = sample_index
        self.transform = transform
        self.name = name

class ITDSpec:
    def __init__(
        self,
        positions: str | tuple[int, ...] | list[int] | np.ndarray = "all",
        plane: str | tuple[object, ...] | dict[str, object] | None = None,
        index_by: str | tuple[str, ...] = ("subject",),
        position_one_hot: bool = False,
        position_index: bool = False,
        method: str = "threshold",
        output: str = "samples",
        thresh_level: float = -10.0,
        upper_cut_freq: float = 3000.0,
        filter_order: int = 10,
        transform: Callable | None = None,
        name: str | None = None,
    ) -> None:
        self.positions = positions
        self.plane = plane
        self.index_by = index_by
        self.position_one_hot = position_one_hot
        self.position_index = position_index
        self.method = method
        self.output = output
        self.thresh_level = thresh_level
        self.upper_cut_freq = upper_cut_freq
        self.filter_order = filter_order
        self.transform = transform
        self.name = name

class ILDSpec:
    def __init__(
        self,
        positions: str | tuple[int, ...] | list[int] | np.ndarray = "all",
        plane: str | tuple[object, ...] | dict[str, object] | None = None,
        index_by: str | tuple[str, ...] = ("subject",),
        position_one_hot: bool = False,
        position_index: bool = False,
        frequency_one_hot: bool = False,
        frequency_index: bool = False,
        mode: str = "broad-band",
        output: str = "db",
        fft_length: int | None = None,
        epsilon: float = 1e-12,
        transform: Callable | None = None,
        name: str | None = None,
    ) -> None:
        self.positions = positions
        self.plane = plane
        self.index_by = index_by
        self.position_one_hot = position_one_hot
        self.position_index = position_index
        self.frequency_one_hot = frequency_one_hot
        self.frequency_index = frequency_index
        self.mode = mode
        self.output = output
        self.fft_length = fft_length
        self.epsilon = epsilon
        self.transform = transform
        self.name = name

class SHSpec:
    def __init__(
        self,
        sh_order: int,
        ears: str | tuple[str, ...] = "both",
        index_by: str | tuple[str, ...] = ("subject",),
        ear_one_hot: bool = False,
        ear_index: bool = False,
        frequency_one_hot: bool = False,
        frequency_index: bool = False,
        epsilon: float = 1e-6,
        transform: Callable | None = None,
        name: str | None = None,
    ) -> None:
        self.sh_order = sh_order
        self.ears = ears
        self.index_by = index_by
        self.ear_one_hot = ear_one_hot
        self.ear_index = ear_index
        self.frequency_one_hot = frequency_one_hot
        self.frequency_index = frequency_index
        self.epsilon = epsilon
        self.transform = transform
        self.name = name

class MeshSpec:
    def __init__(
        self,
        path: str | Path | None = None,
        extensions: tuple[str, ...] | None = None,
        transform: Callable | None = None,
        name: str | None = None,
    ) -> None:
        self.path = path
        self.extensions = extensions
        self.transform = transform
        self.name = name

class AnthropometrySpec:
    def __init__(
        self,
        path: str | Path | None = None,
        extensions: tuple[str, ...] | None = None,
        exclude_row: int | Sequence[int] | None = None,
        exclude_column: int | Sequence[int] | None = None,
        accessed_by: str = "row",
        grouped_by: str | tuple[str, ...] = ("subject",),
        ear: str | None = "",
        ear_one_hot: bool = False,
        ear_index: bool = False,
        transform: Callable | None = None,
        name: str | None = None,
    ) -> None:
        if isinstance(grouped_by, str):
            grouped_by_normalized = (str(grouped_by).strip().lower(),)
            if grouped_by_normalized[0] == "subject":
                grouped_by_normalized = ("subject",)
            elif grouped_by_normalized[0].startswith("subject-"):
                grouped_by_normalized = tuple(
                    part
                    for part in grouped_by_normalized[0].split("-")
                    if part != ""
                )
            else:
                grouped_by_normalized = tuple(grouped_by_normalized)
        else:
            grouped_by_normalized = tuple(str(value).strip().lower() for value in grouped_by)
        if grouped_by_normalized not in {("subject",), ("subject", "ear")}:
            raise ValueError("AnthropometrySpec grouped_by must be ('subject',) or ('subject', 'ear')")
        if ear is None or str(ear).strip() == "":
            ear_value = None
        else:
            ear_value = str(ear).strip().lower()
            if ear_value == "both":
                ear_value = "both"
            elif ear_value not in {"left", "right"}:
                raise ValueError("AnthropometrySpec ear must be None, 'both', 'left', or 'right'")
        self.exclude_row = exclude_row
        self.extensions = extensions
        self.exclude_column = exclude_column
        self.grouped_by = grouped_by_normalized
        self.ear = ear_value
        self.accessed_by = str(accessed_by).strip().lower()
        if self.accessed_by not in {"row", "column"}:
            raise ValueError("AnthropometrySpec accessed_by must be 'row' or 'column'")
        self.ear_one_hot = ear_one_hot
        self.ear_index = ear_index
        self.path = path
        self.transform = transform
        self.name = name

class ImageSpec:
    def __init__(
        self,
        path: str | Path | None = None,
        grouped_by: str | tuple[str, ...] = ("subject",),
        extensions: tuple[str, ...] | None = None,
        ear_one_hot: bool = False,
        ear_index: bool = False,
        concatenate: bool = False,
        transform: Callable | None = None,
        name: str | None = None,
    ) -> None:
        self.path = path
        self.grouped_by = grouped_by
        self.extensions = extensions
        self.ear_one_hot = ear_one_hot
        self.ear_index = ear_index
        self.concatenate = concatenate
        self.transform = transform
        self.name = name

class VideoSpec:
    def __init__(
        self,
        path: str | Path | None = None,
        grouped_by: str | tuple[str, ...] = ("subject",),
        extensions: tuple[str, ...] | None = None,
        ear_one_hot: bool = False,
        ear_index: bool = False,
        transform: Callable | None = None,
        name: str | None = None,
    ) -> None:
        self.path = path
        self.grouped_by = grouped_by
        self.extensions = extensions
        self.ear_one_hot = ear_one_hot
        self.ear_index = ear_index
        self.transform = transform
        self.name = name
