__all__ = [
    "HRTFSpec",
    "ITDSpec",
    "ILDSpec",
    "SHSpec",
    "MeshSpec",
    "AnthropometrySpec",
    "MetadataSpec",
    "ImageSpec",
    "VideoSpec",
    "HRTFTransform",
    "HUTUBS",
    "SONICOM",
]


def __getattr__(name: str):
    if name == "HRTFSpec":
        from .specs import HRTFSpec
        return HRTFSpec
    if name == "ITDSpec":
        from .specs import ITDSpec
        return ITDSpec
    if name == "ILDSpec":
        from .specs import ILDSpec
        return ILDSpec
    if name == "SHSpec":
        from .specs import SHSpec
        return SHSpec
    if name == "MeshSpec":
        from .specs import MeshSpec
        return MeshSpec
    if name == "AnthropometrySpec":
        from .specs import AnthropometrySpec
        return AnthropometrySpec
    if name == "MetadataSpec":
        from .specs import MetadataSpec
        return MetadataSpec
    if name == "ImageSpec":
        from .specs import ImageSpec
        return ImageSpec
    if name == "VideoSpec":
        from .specs import VideoSpec
        return VideoSpec
    if name == "HRTFTransform":
        from .transforms import HRTFTransform
        return HRTFTransform
    if name == "HUTUBS":
        from .hutubs import HUTUBS
        return HUTUBS
    if name == "SONICOM":
        from .sonicom import SONICOM
        return SONICOM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
