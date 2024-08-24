from llamaRun.models.mixtral.block import WrappedMixtralBlock
from llamaRun.models.mixtral.config import DistributedMixtralConfig
from llamaRun.models.mixtral.model import (
    DistributedMixtralForCausalLM,
    DistributedMixtralForSequenceClassification,
    DistributedMixtralModel,
)
from llamaRun.utils.auto_config import register_model_classes

register_model_classes(
    config=DistributedMixtralConfig,
    model=DistributedMixtralModel,
    model_for_causal_lm=DistributedMixtralForCausalLM,
    model_for_sequence_classification=DistributedMixtralForSequenceClassification,
)
