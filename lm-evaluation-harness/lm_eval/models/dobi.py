"""lm-eval model wrappers for the two DOBI distillation checkpoints (discrete
mask-token diffusion and continuous flow matching).

Both checkpoint types are saved by utils.utils.save_hf_model as: the model's own
PretrainedConfig (model_type = "diffusion_excitation" / "flow_excitation") plus a
safetensors file holding ONLY the trainable tensors (the frozen student / teacher-head
/ teacher-embedding weights are re-derived from the base/teacher model ids recorded in
that config, not saved). Loading therefore means: rebuild the model from the saved
config (which re-loads the frozen pieces), then load_state_dict the saved trainable
tensors on top.

These wrappers build that model themselves and hand the finished `torch.nn.Module`
into `HFLM.__init__` via overridden `_get_config`/`_create_tokenizer`/`_create_model`
hooks, so all of HFLM's batching, tokenization, and generation machinery
(generate_until, tok_encode, Collator-based batching, etc.) is reused unmodified.
`huggingface.py` itself is NOT edited.

Registered under:
    --model dobi-diffusion   (DiffusionLM)
    --model dobi-flow        (FlowLM)

Both take a `pretrained` argument that is a path to a saved checkpoint directory (the
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

from components.diffusion_model import DiffusionModel, DiffusionConfig  # noqa: E402
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


def _build_diffusion_model(checkpoint_dir, device):
    config = DiffusionConfig.from_pretrained(checkpoint_dir)
    model = DiffusionModel(config)
    _load_trainable_state(model, checkpoint_dir)
    model = model.to(device)
    model.eval()
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


class _DobiLMMixin:
    """Shared checkpoint-loading behavior for both DOBI model types.

    Overrides `_get_config` (our config's model_type isn't registered with
    transformers.AutoConfig, so the generic lookup would raise), `_create_tokenizer`
    (load the tokenizer save_hf_model wrote alongside the checkpoint), and
    `_create_model` (build+load the DOBI model instead of a generic
    AutoModelForCausalLM). These three run — in this order — inside HFLM.__init__
    whenever `pretrained` is a string, which is what both subclasses pass through.

    `generate_until` is inherited unchanged: both DOBI models implement `.generate()`
    with a standard `generate(input_ids, attention_mask, stopping_criteria, ...)`
    signature, matching what `HFLM._model_generate` already calls.
    """

    def _get_config(self, pretrained, **kwargs):
        self._config = self._config_cls.from_pretrained(pretrained)

    def _create_tokenizer(self, pretrained, tokenizer, **kwargs):
        import transformers
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained, trust_remote_code=True
        )

    def _create_model(self, pretrained, **kwargs):
        self._model = self._build_dobi_model(pretrained, self._device)


@register_model("dobi-diffusion")
class DiffusionLM(_DobiLMMixin, HFLM):
    """Evaluate a DOBI discrete mask-token diffusion checkpoint.

    `pretrained` must be a directory written by utils.utils.save_hf_model for a
    DiffusionModel (contains diffusion_excitation config.json + model.safetensors +
    tokenizer files).

    Loglikelihood scoring (MMLU-style, forward-only) needs its own
    `_loglikelihood_tokens`/`_model_call`: DiffusionModel.forward is token-at-slot, not
    next-token, so it needs to know exactly which input_ids positions are the scored
    continuation (`cont_spans`) to mask ONLY those and keep the context real — the base
    HFLM instead feeds `(context+continuation)[:-1]` and only learns the continuation's
    position in the OUTPUT after the call (too late for a model that must decide what
    to mask before its forward pass). See diffusion_model.py's "Eval / lm-eval
    loglikelihood path" for the corresponding internal next-token-shift + cont_spans
    handling — this override is upstream of that, on the harness side.
    """

    _config_cls = DiffusionConfig

    def _build_dobi_model(self, pretrained, device):
        return _build_diffusion_model(pretrained, device)

    def __init__(self, pretrained, num_sampling_steps=None, **kwargs):
        # The MaskUNet's positional embedding is capped at config.max_seq_len; force
        # lm-eval to respect that budget instead of falling back to the (much larger)
        # base model's max_position_embeddings. One extra [MASK] slot is appended
        # internally by DiffusionModel.forward's loglikelihood path, so cap one below
        # the true max.
        kwargs.setdefault("truncation", True)
        super().__init__(pretrained=pretrained, **kwargs)

        unet_cap = self.model.unet.max_seq_len
        safe_cap = unet_cap - 1
        if self._max_length is None or self._max_length > safe_cap:
            self._max_length = safe_cap

        if num_sampling_steps is not None:
            self.model.num_sampling_steps = int(num_sampling_steps)

    def _model_call(self, inps, attn_mask=None, labels=None, cont_spans=None):
        with torch.no_grad():
            return self.model(input_ids=inps, attention_mask=attn_mask, cont_spans=cont_spans).logits

    def _loglikelihood_tokens(self, requests, disable_tqdm=False, override_bs=None):
        # Mirrors HFLM._loglikelihood_tokens (huggingface.py) with one change: the
        # model input is the FULL (context+continuation) sequence — not [:-1] — and we
        # additionally compute+pass `cont_spans` (in that same, un-shifted input_ids
        # space) into _model_call. DiffusionModel.forward performs its own next-token
        # shift internally and returns logits already aligned to the standard
        # next-token convention, so the OUTPUT-side indexing below (mirroring
        # _select_cont_toks for the causal backend) is otherwise identical to base.
        import torch.nn.functional as F
        from tqdm import tqdm
        from lm_eval.models.utils_hf import pad_and_concat
        from lm_eval.models.utils import Collator

        res = []

        def _collate(req):
            toks = req[1] + req[2]
            return -len(toks), tuple(toks)

        re_ord = Collator(requests, sort_fn=_collate, group_by=None)
        batch_size = (
            self.batch_size if self.batch_size != "auto"
            else (override_bs if override_bs is not None else 0)
        )
        chunks = re_ord.get_batched(n=batch_size, batch_fn=None)

        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running loglikelihood requests (dobi-diffusion)",
        )
        for chunk in chunks:
            inps, cont_toks_list, inplens, cont_spans = [], [], [], []
            padding_len_inp = None

            for _, context_enc, continuation_enc in chunk:
                assert len(context_enc) > 0
                assert len(continuation_enc) > 0
                assert len(continuation_enc) <= self.max_length

                total_length = len(context_enc) + len(continuation_enc)
                if total_length > self.max_length:
                    context_enc = context_enc[-(self.max_length - len(continuation_enc)):]

                full = context_enc + continuation_enc
                inp = torch.tensor(full[-self.max_length:], dtype=torch.long, device=self.device)
                (inplen,) = inp.shape

                cont_len = len(continuation_enc)
                cont_spans.append((inplen - cont_len, inplen))  # input_ids-space span

                padding_len_inp = max(padding_len_inp, inplen) if padding_len_inp is not None else inplen
                inps.append(inp)
                cont_toks_list.append(continuation_enc)
                inplens.append(inplen)

            batched_inps = pad_and_concat(padding_len_inp, inps, padding_side="right")
            # MaskUNet's self-attention is FULLY BIDIRECTIONAL (unlike a causal LM), so
            # right-padding must be masked explicitly — otherwise padded positions leak
            # into every real position's prediction, not just the ones after them.
            attn_mask = torch.zeros(len(inps), padding_len_inp, dtype=torch.long, device=self.device)
            for i, l in enumerate(inplens):
                attn_mask[i, :l] = 1

            multi_logits = F.log_softmax(
                self._model_call(batched_inps, attn_mask=attn_mask, cont_spans=cont_spans),
                dim=-1,
                dtype=self.softmax_dtype,
            )  # [batch, padding_len_inp, vocab] — already next-token-shifted by forward()

            for (request_str, ctx_tokens, _), logits, cont_toks, (start, end) in zip(
                chunk, multi_logits, cont_toks_list, cont_spans, strict=True
            ):
                logits = logits[start:end].unsqueeze(0)  # [1, contlen, vocab]

                greedy_tokens = logits.argmax(dim=-1)
                cont_toks_t = torch.tensor(cont_toks, dtype=torch.long, device=self.device).unsqueeze(0)
                max_equal = (greedy_tokens == cont_toks_t).all()

                logits = torch.gather(logits, 2, cont_toks_t.unsqueeze(-1)).squeeze(-1)
                answer = (float(logits.sum()), bool(max_equal))
                res.append(answer)

                if request_str is not None:
                    self.cache_hook.add_partial("loglikelihood", request_str, answer)
                pbar.update(1)

        pbar.close()
        return re_ord.get_original(res)


@register_model("dobi-flow")
class FlowLM(_DobiLMMixin, HFLM):
    """Evaluate a DOBI flow-matching checkpoint.

    `pretrained` must be a directory written by utils.utils.save_hf_model for a
    FlowModel (contains flow_excitation config.json + model.safetensors + tokenizer
    files).

    Unlike DiffusionModel, FlowModel.forward already follows the standard causal
    next-token convention (logits[t] predicts token t+1, read out through the
    teacher's lm_head on the student+flow combined hidden state) — so the base HFLM
    `_model_call`/`_loglikelihood_tokens` work unmodified; only checkpoint loading
    (via _DobiLMMixin) differs from a plain HF model.
    """

    _config_cls = FlowConfig

    def _build_dobi_model(self, pretrained, device):
        return _build_flow_model(pretrained, device)

    def __init__(self, pretrained, num_flow_steps=None, **kwargs):
        super().__init__(pretrained=pretrained, **kwargs)
        if num_flow_steps is not None:
            self.model.num_steps = int(num_flow_steps)
