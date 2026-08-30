from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("efm")
except PackageNotFoundError:
    __version__ = "0.0.0.dev"

from efm.aggregators import (
    AGGREGATORS,
    CLOSED_FORM,
    EvidenceAggregator,
    GaussianTemplate,
    HyperplaneArrangement,
    StateCoupledRecurrence,
    StudentTTemplate,
    build_aggregator,
    normalize_aggregator_name,
)
from efm.estimator import EFMClassifier, EFMRegressor
from efm.explain import RuleExplainer
from efm.models import EFM
from efm.utils import set_seed

__all__ = [
    "EFM",
    "EFMClassifier",
    "EFMRegressor",
    "EvidenceAggregator",
    "GaussianTemplate",
    "StudentTTemplate",
    "HyperplaneArrangement",
    "StateCoupledRecurrence",
    "AGGREGATORS",
    "CLOSED_FORM",
    "build_aggregator",
    "normalize_aggregator_name",
    "RuleExplainer",
    "set_seed",
    "__version__",
]
