"""lm-eval model wrapper for the DOBI flow-matching distillation checkpoint.

The checkpoint is saved by utils.utils.save_hf_model as: the model's own
PretrainedConfig (model_type = "flow_excitation") plus a safetensors file holding
ONLY the trainable tensors (the frozen student / teacher-head weights are re-derived
from the base/teacher model ids recorded in that config, not saved). Loading therefore
means: rebuild the model from the saved config (which re-loads the frozen pieces),
then load_state_dict the saved trainable tensors on top.

This wrapper builds that model itself and hands the finished `torch.nn.Module`
into `HFLM.__init__` via overridden `_get_config`/`_create_tokenizer`/`_create_model`
hooks, so all of HFLM's batching, tokenization, and generation machinery
(generate_until, tok_encode, Collator-based batching, etc.) is reused unmodified.
`huggingface.py` itself is NOT edited.

Registered under:
    --model dobi-flow

Takes a `pretrained` argument that is a path to a saved checkpoint directory (the
one written by save_hf_model), not a bare HF hub id.
"""

import os
import sys

import torch
from safetensors.torch import load_file

from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM

# The DOBI components package lives at the repo root, one level up from
# lm-evaluation-harness/. Make it importable regardless of cwd at eval time.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from components.flow_model import FlowModel, FlowConfig  # noqa: E402


def _load_trainable_state(model, checkpoint_dir):
    state_path = os.path.join(checkpoint_dir, "model.safetensors")
    state_dict = load_file(state_path)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    # `missing` is expected to be exactly the frozen params (never saved); only fail
    # loudly on tensors the checkpoint has that the freshly-built model doesn't.
    if unexpected:
        raise RuntimeError(
            f"Checkpoint at {checkpoint_dir} has unexpected tensors not present in "
            f"a freshly constructed model: {unexpected}"
        )
    saved = set(state_dict.keys())
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    if not trainable.issubset(saved):
        raise RuntimeError(
            f"Checkpoint at {checkpoint_dir} is missing trainable tensors: "
            f"{trainable - saved}"
        )
    return model


def _build_flow_model(checkpoint_dir, device):
    config = FlowConfig.from_pretrained(checkpoint_dir)
    # FlowModel re-derives its own frozen teacher lm_head from config.teacher_model
    # when no live teacher is handed in (see FlowModel._load_teacher_head).
    model = FlowModel(config, teacher_lm_head=None)
    _load_trainable_state(model, checkpoint_dir)
    model = model.to(device)
    model.eval()
    return model


@register_model("dobi-flow")
class FlowLM(HFLM):
    """Evaluate a DOBI flow-matching checkpoint.

    `pretrained` must be a directory written by utils.utils.save_hf_model for a
    FlowModel (contains flow_excitation config.json + model.safetensors + tokenizer
    files).

    FlowModel.forward follows the standard causal next-token convention (logits[t]
    predicts token t+1, read out through the teacher's lm_head on the student+flow
    combined hidden state) — so the base HFLM `_model_call`/`_loglikelihood_tokens`
    work unmodified; only checkpoint loading (via the overrides below) differs from
    a plain HF model.

    Overrides `_get_config` (our config's model_type isn't registered with
    transformers.AutoConfig, so the generic lookup would raise), `_create_tokenizer`
    (load the tokenizer save_hf_model wrote alongside the checkpoint), and
    `_create_model` (build+load the FlowModel instead of a generic
    AutoModelForCausalLM). These three run — in this order — inside HFLM.__init__
    whenever `pretrained` is a string, which is what this subclass passes through.

    `generate_until` is inherited unchanged: FlowModel implements `.generate()` with
    a standard `generate(input_ids, attention_mask, stopping_criteria, ...)`
    signature, matching what `HFLM._model_generate` already calls.
    """

    def _get_config(self, pretrained, **kwargs):
        self._config = FlowConfig.from_pretrained(pretrained)

    def _create_tokenizer(self, pretrained, tokenizer, **kwargs):
        import transformers
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained, trust_remote_code=True
        )

    def _create_model(self, pretrained, **kwargs):
        self._model = _build_flow_model(pretrained, self._device)

    def __init__(self, pretrained, num_flow_steps=None, **kwargs):
        super().__init__(pretrained=pretrained, **kwargs)
        if num_flow_steps is not None:
            self.model.num_steps = int(num_flow_steps)
