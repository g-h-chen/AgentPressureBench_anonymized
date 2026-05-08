"""Dataset construction package."""

from importlib import import_module

__all__ = [
    "build_credit_dataset",
    "build_medical_dataset",
    "build_tabular_dataset",
    "build_text_dataset",
    "verify_credit_side_channels",
    "verify_medical_side_channels",
    "verify_side_channels",
]


_EXPORTS = {
    "build_credit_dataset": ("credit", "build_credit_dataset"),
    "build_medical_dataset": ("medical", "build_medical_dataset"),
    "build_tabular_dataset": ("tabular", "build_tabular_dataset"),
    "build_text_dataset": ("text", "build_text_dataset"),
    "verify_credit_side_channels": ("credit", "verify_credit_side_channels"),
    "verify_medical_side_channels": ("medical", "verify_medical_side_channels"),
    "verify_side_channels": ("common", "verify_side_channels"),
}


def __getattr__(name: str):
    """Lazily load dataset builders so package import stays lightweight."""
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _EXPORTS[name]
    module = import_module(f"{__name__}.{module_name}")
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
