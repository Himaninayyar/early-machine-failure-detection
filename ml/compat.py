"""Compare the library versions a model set was trained with (meta.json "libraries") with the ones installed now.
Python itself is not compared (the project supports 3.12+); only the libraries that the saved models depend on."""
import sys
from importlib import metadata

_PACKAGES = {"scikit-learn": "scikit-learn", "xgboost": "xgboost", "imbalanced-learn": "imbalanced-learn", "numpy": "numpy"}


def installed_versions():
    out = {"python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"}
    for key, dist in _PACKAGES.items():
        try:
            out[key] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            out[key] = None
    return out


def library_mismatches(trained, installed=None):
    """Human-readable warnings, one per library whose version differs from training."""
    installed = installed or installed_versions()
    warnings = []
    for lib, was in (trained or {}).items():
        now = installed.get(lib)
        if lib == "python":
            continue
        if was != now:
            warnings.append(f"{lib} {now or 'not installed'} is installed, but the models were trained with {was}.")
    if warnings:
        warnings.append("Models usually still load, but a changed library can alter results or break loading. "
                        "To get exactly the training versions: pip install -r requirements-lock.txt. Or retrain: python -m ml.train")
    return warnings
