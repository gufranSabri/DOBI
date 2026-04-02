import peft
from copy import deepcopy
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, PreTrainedModel, PretrainedConfig, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

from components.flownet import FlowNet

class F2L_Config(PretrainedConfig):
    model_type = "excitation"

    def __init__(self, base_model: str = "", teacher_model: str = "", lorify: bool = False, num_flow_steps: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.base_model = base_model
        self.teacher_model = teacher_model
        self.lorify = lorify
        self.num_flow_steps = num_flow_steps

class F2L(PreTrainedModel, GenerationMixin):
    config_class = F2L_Config
    main_input_name = "input_ids"

    def __init__(self, config: F2L_Config):
        super().__init__(config)

        self.num_steps = config.num_flow_steps

        self.teacher = AutoModelForCausalLM.from_pretrained(
            config.teacher_model,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.head = deepcopy(self.teacher.lm_head).to(torch.float32)  # match model dtype
        for param in self.head.parameters():
            param.requires_grad = False

        self.flownet = FlowNet(hidden_dim=self.teacher.config.hidden_size, d_model=512, num_layers=4, dropout=0.0)

        d_large = self.teacher.config.hidden_size
        self.d_large = d_large
        del self.teacher

        self.model = AutoModelForCausalLM.from_pretrained(
            config.base_model,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )

        if config.lorify:
            lora_config = peft.LoraConfig(r=16, lora_alpha=32, lora_dropout=0.1, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
            self.model = peft.get_peft_model(self.model, lora_config)
        else:
            for param in self.model.parameters():
                param.requires_grad = False

        d_small = self.model.config.hidden_size
        self.projector = nn.Sequential(
            nn.Linear(d_small, d_large),
            nn.GELU(),
            nn.Linear(d_large, d_large)
        )

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

    def get_time_weight(self, t_int: torch.Tensor) -> torch.Tensor:
        # t_int: [B] integers in [0, 1000]
        # w_d(t) = wθd(t) - wθd(0), which is 0 at t=0 and grows with t
        # Simplest: w_d(t) = t/1000, shape [B, 1, 1] for broadcasting over [B, T, D]
        return (t_int.float() / 1000.0).view(-1, 1, 1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values=None,
        teacher_embeddings=None, 
        **kwargs,
    ):
        # 1. Base Student Inference
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        
        # Project to teacher dimension
        h_student = self.projector(outputs.hidden_states[-1]) 
        
        if self.training:
            B, T, D = h_student.shape
            
            # Sample time t ~ U([0, 1])
            t = torch.rand(B, device=h_student.device)
            t_int = (t * 1000).long()
            t_expand = t.view(B, 1, 1).expand(B, T, 1)

            # Flow endpoints: x0 (student) -> x1 (Target Residual)
            x0 = h_student
            x1 = teacher_embeddings - h_student
            
            xt = t_expand * x1 + (1 - t_expand) * x0
            target_velocity = x1 - x0
            predicted_velocity = self.flownet(xt, t_int, attention_mask, context=h_student)

            fm_loss_raw = F.mse_loss(predicted_velocity, target_velocity, reduction='none')
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