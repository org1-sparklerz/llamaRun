from llamaRun.models.llama.block import WrappedLlamaBlock
from llamaRun.models.llama.config import DistributedLlamaConfig
from llamaRun.models.llama.model import (
    DistributedLlamaForCausalLM,
    DistributedLlamaForSequenceClassification,
    DistributedLlamaModel,
)
from llamaRun.models.llama.speculative_model import DistributedLlamaForSpeculativeGeneration
from llamaRun.utils.auto_config import register_model_classes

register_model_classes(
    config=DistributedLlamaConfig,
    model=DistributedLlamaModel,
    model_for_causal_lm=DistributedLlamaForCausalLM,
    model_for_speculative=DistributedLlamaForSpeculativeGeneration,
    model_for_sequence_classification=DistributedLlamaForSequenceClassification,
)
