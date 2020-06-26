from ._version import get_versions
from __psvWave_cpp import fdModel

__version__ = get_versions()["version"]
__full_revisionid__ = get_versions()["full-revisionid"]
__version_date__ = get_versions()["date"]
del get_versions
