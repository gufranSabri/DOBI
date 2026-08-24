"""FlowModel — continuous flow-matching distiller.

A frozen student LM produces a final hidden state, projected up into the teacher's
hidden-dim space. A FlowNet (DiT-style transformer) learns the velocity field of a
straight-line path between the (projected) student hidden state and the teacher's
hidden state; walking that field approximates the teacher's residual without ever
running the teacher at inference time. Logits are read out through the teacher's own
(frozen, copied) lm_head applied to the combined (student + predicted residual) hidden.

Compared with the discrete DiffusionModel:
  * The diffusion variable is a CONTINUOUS hidden-state vector, not token ids.
  * Readout is the teacher's frozen lm_head on a continuous vector, not a tied
    embedding matmul.
  * At inference the model integrates an ODE (K Euler steps) rather than iteratively
    unmasking a discrete canvas.
"""

from copy import deepcopy
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, PreTrainedModel, PretrainedConfig, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

from components.flownet import FlowNet


class FlowConfig(PretrainedConfig):
    model_type = "flow_excitation"

    def __init__(
        self,
        base_model: str = "",
        teacher_model: str = "",
        teacher_hidden_size: int = 0,
        flownet_d_model: int = 512,
        flownet_layers: int = 4,
        flownet_heads: int = 8,
        num_flow_steps: int = 10,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_model = base_model
        self.teacher_model = teacher_model
        self.teacher_hidden_size = teacher_hidden_size
        self.flownet_d_model = flownet_d_model
        self.flownet_layers = flownet_layers
        self.flownet_heads = flownet_heads
        self.num_flow_steps = num_flow_steps


class FlowModel(PreTrainedModel, GenerationMixin):
    config_class = FlowConfig
    main_input_name = "input_ids"

    def __init__(self, config: FlowConfig, teacher_lm_head: Optional[nn.Module] = None):
        super().__init__(config)

        self.num_steps = config.num_flow_steps
        d_large = config.teacher_hidden_size
        assert d_large > 0, "FlowConfig.teacher_hidden_size must be set before building FlowModel."
        self.d_large = d_large

        # Readout head: a frozen COPY of the teacher's lm_head (never the teacher itself
        # — the teacher is loaded once by the caller for its own forward pass and is not
        # owned by this module). Reconstructed at load time via _load_teacher_head if not
        # passed directly (see from_pretrained_with_teacher below).
        if teacher_lm_head is not None:
            self.head = deepcopy(teacher_lm_head).to(torch.float32)
        else:
            self.head = self._load_teacher_head(config.teacher_model, d_large)
        for p in self.head.parameters():
            p.requires_grad = False

        self.flownet = FlowNet(
            hidden_dim=d_large,
            d_model=config.flownet_d_model,
            num_layers=config.flownet_layers,
            num_heads=config.flownet_heads,
            dropout=0.0,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            config.base_model,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )
        for param in self.model.parameters():
            param.requires_grad = False

        d_small = self.model.config.hidden_size
        self.projector = nn.Sequential(
            nn.Linear(d_small, d_large),
            nn.GELU(),
            nn.Linear(d_large, d_large),
        )

    @staticmethod
    def _load_teacher_head(teacher_model_id, expected_hidden_size):
        """Recover just the teacher's lm_head without holding the full teacher in
        memory for longer than one call — used when reloading a saved checkpoint,
        where the caller has not already loaded a live teacher to copy from."""
        teacher = AutoModelForCausalLM.from_pretrained(
            teacher_model_id,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )
        assert teacher.config.hidden_size == expected_hidden_size, (
            f"Teacher hidden size changed ({teacher.config.hidden_size}) vs the "
            f"checkpoint's recorded value ({expected_hidden_size})."
        )
        head = deepcopy(teacher.lm_head)
        del teacher
        return head

    def can_generate(self) -> bool:
        return True

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values=None,
        **kwargs,
    ):
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
            # attention_mask stays FULL length — do NOT slice it

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values=None,
        teacher_embeddings: Optional[torch.Tensor] = None,
        cont_spans=None,   # accepted for interface parity with DiffusionModel; unused
        return_dict: bool = True,
        **kwargs,
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )

        h_student = self.projector(outputs.hidden_states[-1])

        if self.training:
            assert teacher_embeddings is not None, "Training FlowModel requires teacher_embeddings."
            B, T, D = h_student.shape

            t = torch.rand(B, device=h_student.device)
            t_int = (t * 1000).long()
            t_expand = t.view(B, 1, 1).expand(B, T, 1)

            x0 = h_student
            x1 = teacher_embeddings - h_student

            xt = t_expand * x1 + (1 - t_expand) * x0
            target_velocity = x1 - x0
            predicted_velocity = self.flownet(xt, t_int, attention_mask, context=h_student)

            fm_loss_raw = nn.functional.mse_loss(predicted_velocity, target_velocity, reduction="none")
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).to(fm_loss_raw.dtype)
                fm_loss = (fm_loss_raw * mask).sum() / (mask.sum() * D)
            else:
                fm_loss = fm_loss_raw.mean()

            combined = h_student + x1
            logits = self.head(combined)

            return CausalLMOutputWithPast(
                loss=fm_loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=combined,
            )

        else:
            B, T, _ = h_student.shape
            xt = h_student
            dt = 1.0 / self.num_steps

            for i in range(self.num_steps):
                t_val = i / self.num_steps
                t_int = torch.full((B,), int(t_val * 1000), device=h_student.device, dtype=torch.long)

                v = self.flownet(xt, t_int, attention_mask, context=h_student)
                xt = xt + v * dt

            combined = h_student + xt
            logits = self.head(combined)

            return CausalLMOutputWithPast(
                loss=None,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=combined,
            )
