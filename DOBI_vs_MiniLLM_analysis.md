# DOBI (flow) vs. MiniLLM — why the baseline isn't being beaten

**Scope.** Compared `components/flow_model.py` + `components/flownet.py` + `components/trainers.py` (current flow-matching student) against `old/minillm/finetune.py` + `old/minillm/minillm/{trainer,losses,utils}.py` (the reference KD pipeline), plus your own prior prototype in `old/DOBI-flow/components/f2l.py`. Baseline = frozen Qwen2.5-1.5B-Instruct student, no flow head at all. Per memory, v2 saturated at top-1-vs-teacher ≈ 0.67 against a frozen-student floor of **0.869** — the untouched base model already beats the trained flow model. That's the failure this report is about.

---

## 1. The one architectural fact that explains most of the gap

**MiniLLM trains the student's own weights.** In `finetune.py::finetune`, `model` *is* the student LM (wrapped in DeepSpeed); `model.backward(loss)` / `model.step()` update every weight, either full fine-tuning or LoRA. The distillation loss (`get_distil_loss`, forward-KL against the teacher) and the LM loss both flow gradients straight into the student transformer. The student's own next-token distribution is what's being optimized, directly.

**DOBI freezes the student and asks a small side-network to fix its output post-hoc.** In `components/flow_model.py:85-86`:

```python
self.model = AutoModelForCausalLM.from_pretrained(config.base_model, ...)
for param in self.model.parameters():
    param.requires_grad = False
```

The only trainable pieces are `projector` (2-layer MLP, ~a few million params) and `flownet` (4-layer transformer, ~30M params at `D_FLOW=512`). Everything these modules can output is a *function of the frozen student's last hidden state* — they can reshape/rotate/denoise that vector, but they cannot inject any information the frozen 1.5B model didn't already put there. If the frozen student's hidden state doesn't linearly (or near-linearly) encode "what the 3B teacher would have said," no amount of flow-matching on top will recover it, because the student itself never adjusted to make that information available.

This is why the frozen-student floor (0.869) beats the trained flow model (0.67): the frozen student's own `lm_head` was *trained end-to-end* to turn its hidden state into good next-token logits. Your `projector → flownet → teacher's lm_head` path re-purposes the teacher's head (never seen this student's hidden-state distribution during its own training) as the readout, going through two freshly-initialized networks that have to learn a mapping the student's own head already solved by construction.

**Bottom line:** this isn't a flow-matching bug, it's a capacity/gradient-path problem. The frozen backbone never learns to *produce* teacher-recoverable representations; only the tiny bolt-on networks are asked to reconstruct them after the fact.

---

## 2. Loss composition: MiniLLM mixes CE + KD on the SAME logits; DOBI's two losses fight

MiniLLM (`finetune.py:270-274`):
```python
loss = (1 - kd_ratio) * lm_loss + kd_ratio * distil_loss
```
Both terms are computed from **the same student forward pass**, on the same logits, with the same gradient path back into one set of weights. They're complementary regularizers on one distribution.

DOBI's `FlowTrainer.compute_loss` (`components/trainers.py`):
```python
loss_flow = student_out.loss           # MSE(predicted_velocity, target_velocity) — a HIDDEN-STATE space objective
loss_kl   = self.kl_fn(s_logits, t_logits, ...)   # a LOGIT-space objective, read out through predicted_hidden
loss = W_FLOW * loss_flow + W_KL * loss_kl
```
`loss_flow` optimizes velocity-field accuracy in the teacher's 1536-dim hidden space. `loss_kl` optimizes the *logits* produced by pushing one Euler step of that same predicted velocity through the frozen teacher head (`flow_model.py:182-183`, `predicted_hidden = x0 + t_expand * predicted_velocity`). These are not obviously aligned objectives: a velocity field can have low MSE in hidden-space while producing logits that are off in exactly the high-magnitude directions that matter for argmax-token agreement (hidden-space MSE weights all 1536 dims equally; token identity lives in a much lower-dimensional, teacher-head-dependent subspace). MiniLLM never has this tension because there's only one output space (logits) being optimized.

Also worth flagging: `t = torch.rand(B)` samples one timestep **per batch element, not per token**, and `predicted_hidden` for the *loss_kl* term is evaluated at that single random `t`, not at `t≈1` (full denoising). Early in training when `t` is drawn near 0, `predicted_hidden ≈ x0` (the student's own untouched projection) and the KL loss is effectively being computed on the student's own hidden state through a head that was never trained for it — a noisy, high-variance training signal for the KL term specifically.

---

## 3. MiniLLM's real innovation (reverse-KL + RL) isn't in your pipeline at all

MiniLLM's headline contribution is **stage 2**: PPO-style RL (`minillm/trainer.py`, `minillm/losses.py`) where the student *generates its own sequences* and is rewarded by `get_rev_kl` (`minillm/utils.py:66-69`):
```python
def get_rev_kl(log_p, log_q, mask):
    log_ratio = (log_p - log_q) * mask
    return log_ratio.float().exp() - 1 - log_ratio     # k3 estimator of reverse KL
```
This is trained **on-policy**: the student samples its own text, and the reverse-KL objective specifically penalizes the student for putting probability mass on tokens the teacher considers implausible (mode-seeking, avoids the "average over teacher's modes" failure of forward-KL that plain CE-style distillation has). This is the mechanism the MiniLLM paper credits for beating standard KD.

DOBI currently only does the equivalent of MiniLLM's **stage-1 SFT+KD warmup** — everything is off-policy, teacher-forced on the fixed dataset, no on-policy sampling/reward loop. `skewed_reverse_kl` (`utils/losses.py:80-95`) applies the mode-seeking *direction* of KL, but not the *on-policy sampling* that makes it effective in MiniLLM — off-policy reverse-KL is known to be a much weaker signal (matches the paper's own ablations, which show stage-1-only KD underperforming their full pipeline).

---

## 4. FlowNet conditioning: v2.1 already diagnosed a starvation issue; check it's actually wired

Per `[[dobi-flow-v21-state]]`, v2 FlowNet got "NO direct student-final-hidden input, only K/V cross-attn through a width-256 trunk," and this was identified as a likely cause of underfitting (FM loss stuck near the trivial-predictor baseline). v2.1 was supposed to add `context=h_student` concatenation. Current `flow_model.py:169` does pass `context=h_student` into every `FlowNet` block, and `flownet.py`'s `FlowBlock.forward` cross-attends `x` (the running flow state) against that context at every layer — so the conditioning fix does appear to be in place. Worth double-checking empirically (log FM loss vs. the trivial-predictor baseline `MSE(0, target_velocity)` early in a fresh run) that this isn't still under-conditioned; the projector's output dimension (1536) and FlowNet's `d_model` (512, per `configs/flow.yaml`) means every cross-attention KV projection is compressing the 1536-dim student signal down to 512 before the query even sees it, which re-introduces a bottleneck even with `context` wired up.

---

## 5. Concrete, faithful-to-flow-matching changes to try

Ranked by expected impact, staying inside "flow matching from student hidden state to teacher hidden state" — none of these require abandoning the idea, but items 1-2 are the ones most likely to actually move the needle, based on the diagnosis above.

1. **Unfreeze the student (or add LoRA on it), the single biggest change.** This is the actual MiniLLM insight applied faithfully: let gradients from `loss_flow`/`loss_kl` flow back through `self.model`, not just through `projector`/`flownet`. Your own `old/DOBI-flow/components/f2l.py:54-59` already had this exact toggle (`config.lorify` → `peft.LoraConfig(...)` on `q_proj/k_proj/v_proj/o_proj`) before it was removed per `[[dobi-lora-removed-multigpu-added]]`. Re-adding LoRA on the student (not full fine-tuning, to keep it cheap) lets the backbone itself learn to produce hidden states the flow head can actually map to the teacher's — instead of asking a frozen, never-adapted representation to already contain that information. This directly targets the §1 diagnosis and costs relatively little extra memory/compute versus full unfreezing.

2. **Train the flow objective on hidden states in the teacher's *pre-head* space that's actually informative for logits, not just any 1536-dim MSE.** Consider adding a small auxiliary term that scores `predicted_hidden` through the frozen teacher head *at a late `t` close to 1* (or the full multi-step estimate, matching inference) rather than relying on a single random-`t` Euler step for the KL loss. This tightens the coupling between "good FM loss" and "good logits" that §2 shows can currently diverge. Concretely: run the same K-step Euler loop used at inference (or a fixed small number of steps, e.g. 2-3) during training too, and compute `loss_kl` off the *final* denoised state, not an intermediate one at a random `t`. This is more expensive (multiple flownet calls per training step) but is a faithful, principled fix rather than a change of approach.

3. **Weight the flow-matching loss by proximity to `t=1`, or switch to a v-prediction / SNR-weighted loss**, so the network isn't spending equal capacity getting the easy near-`x0` region right versus the hard near-`x1` region where logit-relevant detail lives. Standard trick in diffusion/flow literature (also used in your removed diffusion-model code, which had a `mask_schedule` — worth reusing that intuition here even though the diffusion code itself is gone).

4. **Increase `FLOWNET_D_MODEL`** past 512 if compute allows, or at minimum stop compressing the 1536-dim context down to 512 before cross-attention (§4) — e.g. use a wider `d_model` matching `teacher_hidden_size` more closely, or a rank-preserving projection. This is a cheap experiment to rule out the bottleneck theory before investing in item 1.

5. **Add MiniLLM's on-policy reverse-KL stage as a phase 2**, faithfully adapted: after the current flow-matching warmup converges, sample the *flow model's own* generations (via its existing `generate()`/Euler-integration path), score them with `get_rev_kl`-style reward against the teacher, and fine-tune flownet+projector (and student, if unfrozen per #1) with a PPO-style or even simpler REINFORCE-style update. This is the part of MiniLLM your pipeline is currently missing entirely (§3) and is the mechanism the MiniLLM paper attributes most of its gains to — but it's also the most implementation-heavy item, so sequence it after #1-2 confirm the architecture itself is sound.

6. **Sanity-check the FM-loss-vs-trivial-baseline gap on a fresh run before changing anything else.** `[[dobi-flow-v21-state]]` flagged FM loss stuck at ~1.7-1.9 vs a trivial-predictor baseline of ~2.0 for v2 — barely better than predicting zero velocity. Confirm whether v2.1's conditioning fix (context=h_student, now present per §4) actually closed this gap in isolation, before spending compute on #1-3. If FM loss is still near-trivial, the fix in #1 (unfreezing the student) is even more clearly the right first move, since the frozen hidden states may simply not contain a learnable path to the teacher's.

---

## Summary

The core issue is very likely **not** the flow-matching mechanism itself (loss functions, KL variants, and Euler integration all look like faithful, correct ports of standard KD/flow-matching practice — including several 1:1 ports of MiniLLM's own KL implementations). It's that **the student backbone never gets to adapt**, so the flow head is trying to reconstruct teacher-quality representations from a frozen, KD-agnostic hidden state — something MiniLLM never asks of any component, because MiniLLM trains the student directly. Unfreezing (or LoRA-adapting) the student is the highest-leverage, most faithful-to-your-approach change to try first.
