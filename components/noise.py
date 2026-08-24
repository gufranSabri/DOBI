"""Absorbing-state (mask) discrete diffusion: corruption process and schedules.

The diffusion variable is a length-T canvas of token ids. The forward (noising)
process replaces a fraction of the RESPONSE tokens with a special [MASK] id; the
model learns to reverse it (predict the ORIGINAL token AT every masked position).
Context (prompt) positions are the known "provided beginning" and are never masked.

TOKEN-AT-POSITION CONVENTION (important). Mask diffusion recovers "the token that
lives at slot t", not "the next token after t". So the canvas is just `input_ids`
(a masked slot hides input_ids[t]), and the target at slot t is the token that
occupies slot t. This is DIFFERENT from a causal LM's next-token convention:
  * a causal LM's logits[t] predict the token at slot t+1 (stored at index t);
  * the teacher's prediction OF the token at slot t therefore lives at index t-1.
Two helpers below convert between the two conventions:
  * `response_mask_from_labels` recovers the true response-token positions from the
    PRE-SHIFTED label mask the data loader produces (labels[t] == input_ids[t+1]);
  * `teacher_at_slot` rolls a teacher tensor so index t carries the teacher's
    prediction OF slot t (i.e. teacher_x[t] <- teacher_x[t-1]).

A single continuous timestep t in [0, 1] indexes the noise level via a cosine
schedule (MaskGIT / Dream style):

    mask_ratio(t) = cos((1 - t) * pi/2)

so t -> 1 is fully masked (ratio 1) and t -> 0 is clean (ratio 0). Training samples
t ~ U(0, 1) per example; the same schedule, walked backwards in K discrete steps,
drives confidence-based unmasking at sampling time (see MaskUNet-based generate()).
"""

import math

import torch


def response_mask_from_labels(labels):
    """Recover the TRUE response-token positions (in input_ids space) from the data
    loader's PRE-SHIFTED label mask.

    The loader sets labels[t] == input_ids[t+1] on the scored span, so `labels != -100`
    marks indices [len(ctx)-1 .. len(ctx)+len(resp)-1] — the response span shifted LEFT
    by one (it starts at the last context token and ends at the last response token).
    The real response tokens sit at input_ids indices [len(ctx) .. len(ctx)+len(resp)-1],
    i.e. the label mask shifted RIGHT by one. Rolling right by one recovers them.

    Returns (B, T) bool: True exactly on the response tokens in input_ids.
    """
    lab = labels != -100                                  # (B, T) pre-shifted mask
    resp = torch.zeros_like(lab)
    resp[:, 1:] = lab[:, :-1]                             # shift right by one
    return resp


def teacher_at_slot(x):
    """Roll a per-position teacher tensor so index t carries the teacher's prediction OF
    the token at slot t. The teacher predicts slot t from position t-1, so
    x_at_slot[t] = x[t-1]; slot 0 has no predictor and is left as-is (it is a context
    position, never masked/scored). Works for (B, T) ids or (B, T, V) logits.
    """
    out = x.clone()
    out[:, 1:] = x[:, :-1]
    return out


def mask_ratio(t):
    """Cosine mask schedule. t in [0, 1] (scalar or tensor) -> ratio in [0, 1].

    ratio(0) = 0 (clean), ratio(1) = 1 (all masked), monotone increasing in t.
    """
    if isinstance(t, torch.Tensor):
        return torch.cos((1.0 - t) * (math.pi / 2.0)).clamp(0.0, 1.0)
    return max(0.0, min(1.0, math.cos((1.0 - t) * (math.pi / 2.0))))


def num_still_masked(step, num_steps, length):
    """Count of positions that remain masked AFTER unmasking `step` (0-indexed) of
    `num_steps`. Walks the cosine schedule backwards from all-masked to none:

        keep(step) = ceil(length * cos(((step + 1) / num_steps) * pi/2))

    Two guarantees on top of the raw cosine, so no step is wasted with few steps:
      * keep is strictly decreasing — at least one position commits each step
        (keep(step) <= length - (step + 1));
      * keep(num_steps - 1) == 0 — everything is committed by the final step.
    """
    if step >= num_steps - 1:
        return 0
    frac = math.cos(((step + 1) / num_steps) * (math.pi / 2.0))
    keep = int(math.ceil(length * max(0.0, frac)))
    keep = min(keep, length - (step + 1))     # force ≥1 commit per step
    return max(0, keep)


def sample_mask(loss_mask, t, generator=None):
    """Draw the boolean corruption mask for a batch.

    Args:
        loss_mask: (B, T) bool — True on RESPONSE positions eligible for masking.
        t:         (B,) float in [0, 1] — per-example noise level.
        generator: optional torch.Generator for reproducible draws.

    Returns:
        mask_bool: (B, T) bool — True where the token is replaced by [MASK].
                   Always ⊆ loss_mask, and guaranteed ≥1 masked position per example
                   that has any response position (so the denoising loss is never empty).
    """
    B, T = loss_mask.shape
    device = loss_mask.device
    r = mask_ratio(t).to(device).view(B, 1)                        # (B, 1)

    rand = torch.rand(B, T, device=device, generator=generator)
    mask_bool = loss_mask & (rand < r)                             # (B, T)

    # Guarantee at least one masked position per example that has response tokens:
    # if an example's Bernoulli draw produced none, force-mask its lowest-rand
    # response position. Examples with no response positions at all stay empty.
    has_resp = loss_mask.any(dim=1)                                # (B,)
    empty = has_resp & (~mask_bool.any(dim=1))                     # (B,) need a forced mask
    if empty.any():
        # Among response positions pick the smallest rand; non-response -> +inf so
        # they're never chosen.
        pick = torch.where(loss_mask, rand, torch.full_like(rand, float("inf")))
        forced_idx = pick.argmin(dim=1)                           # (B,)
        rows = torch.nonzero(empty, as_tuple=False).squeeze(1)
        mask_bool[rows, forced_idx[rows]] = True

    return mask_bool


def build_canvas(input_ids, mask_bool, mask_id):
    """Assemble the token ids fed to the UNet embedding.

    The canvas IS the real token sequence (token-at-position convention): every slot
    holds input_ids[t], except masked slots which hold mask_id. Context and unmasked
    response slots therefore keep their real token — one consistent meaning of "the
    token at slot t" across the whole canvas.

        masked slot (mask_bool == True) -> mask_id
        every other slot                -> input_ids[t]

    Returns (B, T) long tensor of canvas token ids.
    """
    canvas = input_ids.clone()
    canvas = torch.where(mask_bool, torch.full_like(canvas, mask_id), canvas)
    return canvas
