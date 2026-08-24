
import random
from pathlib import Path

import numpy as np
import torch

from safetensors.torch import save_file
from transformers import AutoTokenizer


def set_rng_state(seed):
    # Python's `random` must be seeded too: utils/data.py shuffles with it, and under
    # DDP every rank builds the dataset independently — unseeded, each rank would end
    # up with a DIFFERENT sample subset/order after the cap, silently desyncing the
    # DistributedSampler's assumption that all ranks hold the same dataset.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_hf_model(model, save_dir, base_model_name):
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Persist only the trained tensors (the UNet + probe, selected by trainability). The
    # tied output head is the FROZEN student embedding — not trainable, not saved; it is
    # reloaded from the base model. There are no normalization buffers to keep (discrete
    # mask diffusion has no continuous coordinate system).
    trainable_names = {
        name for name, param in model.named_parameters() if param.requires_grad
    }
    state_dict = model.state_dict()
    trainable_state = {k: v for k, v in state_dict.items() if k in trainable_names}
    assert len(trainable_state) > 0, (
        "No trainable weights selected for saving — refusing to write an empty "
        "checkpoint. Check that named_parameters() keys match state_dict() keys."
    )
    save_file(trainable_state, save_path / "model.safetensors")

    # model.config is already a DiffusionConfig (model_type = "diffusion_excitation");
    # save it as-is. The reload path (load_excitation_model) reconstructs DiffusionModel
    # from this config alone and re-derives the base Qwen model from config.base_model,
    # so no fields from the base model's own config need to be merged in here. (A prior
    # version merged onto a cloned Qwen2Config, whose class-level model_type = "qwen2"
    # silently overrode any instance-level override at save_pretrained/to_dict time.)
    model.config.save_pretrained(save_path)

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.save_pretrained(save_path)
