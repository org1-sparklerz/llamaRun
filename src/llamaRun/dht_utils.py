import warnings

warnings.warn(
    "llamaRun.dht_utils has been moved to llamaRun.utils.dht. This alias will be removed in Petals 2.2.0+",
    DeprecationWarning,
    stacklevel=2,
)

from llamaRun.utils.dht import *
