import torch
import torch.nn as nn
import peft
from copy import deepcopy
from pathlib import Path
from transformers import AutoModelForCausalLM, PreTrainedModel, PretrainedConfig, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Optional


class LearnableScale(nn.Module):
    def __init__(self, d: int, init_scale: float = 1.0, noise_std: float = 0.01):
        super().__init__()
        init = torch.ones(d) * init_scale
        init = init + torch.randn(d) * noise_std * init_scale  # noise relative to scale
        self.scale = nn.Parameter(init)

    def forward(self, x):
        return x * self.scale

class AlignmentConfig(PretrainedConfig):
    model_type = "excitation"

    def __init__(self, base_model: str = "", teacher_model: str = "", **kwargs):
        super().__init__(**kwargs)
        self.base_model = base_model
        self.teacher_model = teacher_model

class AlignedModel(PreTrainedModel, GenerationMixin):
    config_class = AlignmentConfig
    main_input_name = "input_ids"

    def __init__(self, config: AlignmentConfig):
        super().__init__(config)

        self.teacher = AutoModelForCausalLM.from_pretrained(
            config.teacher_model,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.head = deepcopy(self.teacher.lm_head).to(torch.float32)  # match model dtype
        for param in self.head.parameters():
            param.requires_grad = False

        d_large = self.teacher.config.hidden_size
        self.d_large = d_large
        del self.teacher

        self.model = AutoModelForCausalLM.from_pretrained(
            config.base_model,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )
        for param in self.model.parameters():
            param.requires_grad = False

        d_small = self.model.config.hidden_size
        self.projector = nn.Sequential(
            nn.Linear(d_small, d_large, bias=True),
            LearnableScale(d_large),
        )

    def set_flownet(self, flownet):
        self.flownet = flownet

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

    def process_flow(self, projected, attention_mask=None):
        B, T, D = projected.shape
        x = projected.clone()
        
        num_steps = getattr(self, "num_flow_steps", 5)
        delta_tau = 1.0 / num_steps

        with torch.no_grad():
            for i in range(num_steps):
                tau_val = i / num_steps
                t_int = torch.full(
                    (B,), int(tau_val * 1000),
                    dtype=torch.long,
                    device=x.device,
                )
                v = self.flownet(x, t=t_int, attention_mask=attention_mask)
                x = x + v * delta_tau

        return x

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values=None,
        return_projected: bool = False,
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

        last_hidden = outputs.hidden_states[-1]  # [B, T, d_small]
        last_hidden = last_hidden.to(self.projector[0].weight.dtype)  # cast to projector dtype if needed
        projected   = self.projector(last_hidden) # [B, T, d_large] — all float32 now, no casting needed

        if hasattr(self, "flownet"):
            projected = self.process_flow(projected, attention_mask=attention_mask)

        logits = self.head(projected)        # [B, T, vocab]

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
            )

        if return_projected:
            return {
                "loss": loss,
                "logits": logits,
                "projected_hidden": projected,
                "past_key_values": outputs.past_key_values,
            }

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )