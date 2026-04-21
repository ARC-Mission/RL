# EGMD: Entropy-Gap Modulated Distillation
# Implementation Spec for NeMo RL

## 1. Problem Being Solved

In on-policy self-distillation, the teacher model sees the student's past reasoning
trace as privileged context. This makes the teacher confident everywhere — both at
positions where confidence is justified (routine computation, redundant steps) and
at positions where it is only confident because it already saw how the student
resolved an uncertain reasoning step. Standard reverse KL distills both types of
confidence equally, which suppresses the student's ability to express uncertainty
and explore alternatives on hard problems (epistemic verbalization).

EGMD fixes this by measuring WHERE the teacher's confidence comes from privileged
info vs genuine reasoning structure, and downweighting the loss at positions where
the teacher's advantage is parasitic on having seen the trace.

## 2. Current Setup (What Already Exists)

Framework: NeMo RL
Training: On-policy self-distillation with reverse KL
Teacher context: Refining instruction + student's past reasoning trace
Student context: Problem only (no privileged info)
Both teacher and student share the same model weights (self-distillation)

## 3. What Changes

Only the loss computation changes. Everything else (rollout generation, teacher
forward pass, optimizer, data pipeline) stays the same. You already have both
pi_student and pi_teacher logits at every position — the only addition is computing
entropy from those logits and using it to weight the per-token KL.

## 4. EGMD Loss — Version 1 (No Entropy Bonus)

### Step 1: Compute per-token entropies from logits you already have

    H_student(t) = -sum_v pi_student(v, t) * log pi_student(v, t)
    H_teacher(t) = -sum_v pi_teacher(v, t) * log pi_teacher(v, t)

Implementation note: Use the full softmax distributions you already compute for
the KL. If you're working with top-k logits (e.g., top-k=100 as in SDPO), compute
entropy over that same top-k. The absolute entropy values don't matter — only the
gap matters, and top-k approximation affects both sides equally.

### Step 2: Compute entropy gap

    delta(t) = H_student(t) - H_teacher(t)

Interpretation:
    delta >> 0: Student is uncertain, teacher is confident. Teacher's confidence
                likely comes from having seen the trace. This is where epistemic
                tokens get suppressed under standard reverse KL.
    delta ~ 0:  Both have similar uncertainty. Teacher's structural improvements
                (better phrasing, cleaner steps) are genuine and safe to distill.
    delta < 0:  Student is more confident than teacher. Rare but possible. Treat
                same as delta ~ 0.

### Step 3: Compute per-token weight via sigmoid gating

    w(t) = sigmoid(-beta * (delta(t) - tau))

where:
    beta = sharpness parameter (controls how binary the gating is)
    tau = threshold (positions with delta > tau get downweighted)

When delta(t) >> tau: w(t) -> 0 (suppress distillation here)
When delta(t) << tau: w(t) -> 1 (full distillation here)
When delta(t) = tau:  w(t) = 0.5

### Step 4: Weighted loss

    L_egmd = (1/T) * sum_t w(t) * KL(pi_student(t) || sg[pi_teacher(t)])

That's it. The gradient of L_egmd with respect to theta has reduced contribution
from high-delta positions, so the parameter update direction favors structural
compression over epistemic suppression.

## 5. EGMD Loss — Version 2 (With Entropy Bonus)

Version 1 has a weakness: at high-delta positions, w(t) ~ 0 means no gradient.
But parameter updates from OTHER positions (where w is high) still change shared
weights, which can erode epistemic behavior as a side effect. Version 2 adds an
active counter-force: at high-delta positions, actively reward the student for
maintaining high entropy.

### Loss

    L_egmd_e = (1/T) * sum_t [
        w(t) * KL(pi_student(t) || sg[pi_teacher(t)])
        - alpha * (1 - w(t)) * H_student(t)
    ]

The second term is an entropy bonus, active at high-delta positions (where
1-w(t) is large). Minimizing this loss = minimizing KL where safe + maximizing
student entropy where the teacher's confidence is parasitic.

Written out fully for a single position t:

    l(t) = sum_v pi_student(v, t) * [
        w(t) * (log pi_student(v, t) - log pi_teacher(v, t))
        + alpha * (1 - w(t)) * log pi_student(v, t)
    ]

Which simplifies to:

    l(t) = sum_v pi_student(v, t) * [
        (w(t) + alpha * (1 - w(t))) * log pi_student(v, t)
        - w(t) * log pi_teacher(v, t)
    ]

This is the form you'd implement — one pass over the vocabulary per position.

## 6. Computing tau (Threshold)

tau should adapt to the current batch, not be a fixed constant. Reason: early in
training, the student is far from the teacher everywhere, so delta is large across
the board. A fixed tau would either gate everything (if too low) or nothing (if
too high).

Compute tau as a percentile of delta over the current batch:

    all_deltas = [delta(t) for all tokens t across all examples in the batch]
    tau = percentile(all_deltas, p)

where p = 60 means: 60% of tokens get full or near-full distillation weight,
40% get reduced weight. Adjust p based on how aggressively you want to preserve
epistemic behavior.

Implementation: This is a single torch.quantile call over a 1D tensor of all
delta values in the batch. Compute once per batch, use as a scalar for the
sigmoid.

## 8. NeMo RL Specific Notes

### Where to modify

The loss computation lives in whatever module computes the reverse KL between
student and teacher. In NeMo RL's on-policy distillation pipeline, this is
typically in the policy loss function that takes student and teacher logits as
input. You are NOT modifying:
- The rollout generation (student generates as before)
- The teacher forward pass (teacher sees refining instruction + past trace as before)
- The optimizer or learning rate schedule
- The data pipeline or batching

You ARE modifying:
- The loss function: from standard reverse KL to weighted reverse KL with
  optional entropy bonus

### If using top-k logits

If the teacher only returns top-k logits (common in distillation for memory),
compute entropy over the top-k subset only. The delta will be slightly
underestimated for the student (whose true entropy over full vocab is higher),
but the relative ordering across positions is preserved, which is what matters
for the gating.

### Gradient accumulation

tau should be computed over the effective batch (after gradient accumulation),
not per micro-batch. If this is impractical, computing per micro-batch is fine
as an approximation — tau will be noisier but the sigmoid smooths it out.

### Multi-GPU

If using data parallel, you need an all_gather on the delta tensor before
computing tau, so the percentile is over the global batch. If this is too
expensive, compute per-GPU tau — it's an approximation but works in practice
because the percentile is a robust statistic.

## 9. Hyperparameter Guide

beta (sigmoid sharpness):
    Low (1-2):  Soft gating. Most tokens get intermediate weight. Gentler but
                the distinction between "privileged confidence" and "genuine
                confidence" is blurred.
    Medium (5): Sharp but smooth transition. Recommended starting point.
    High (10+): Near-binary gating. Tokens are either fully distilled or fully
                masked. Can be unstable if tau estimates are noisy.

p (percentile for tau):
    0.5: Half the tokens get reduced weight. Aggressive epistemic preservation.
    0.6: Recommended starting point.
    0.7: Most tokens get distilled. Less preservation, more compression.
    0.8+: Approaching standard reverse KL. Only the most extreme entropy gaps
          get protected.

alpha (entropy bonus, Version 2 only):
    0.0:  Equivalent to Version 1.
    0.05: Very mild preservation. Start here if worried about instability.
    0.1:  Recommended starting point.
    0.3+: Strong entropy preservation. Risk: model may resist compression even
          where it's safe. Monitor response length — if it stops decreasing on
          easy problems, alpha is too high.

## 10. What to Log (Diagnostics)

These are critical for debugging. Log per training step:

1. mean_w: Average w(t) over valid tokens in the batch.
   Expected: 0.4-0.7 and roughly stable.
   If ~1.0: teacher isn't adding privileged info (delta is small everywhere).
            This means EGMD reduces to standard reverse KL. Check that the
            teacher actually sees the student trace.
   If ~0.0: almost all teacher confidence is from the trace. Very little
            distillation is happening. Consider reducing the richness of the
            teacher context or lowering p.

2. tau: The computed threshold per step.
   Expected: Positive and roughly stable after initial steps.
   If negative: Student is generally more confident than teacher (unusual).
                Check that teacher and student contexts are set up correctly.

3. epistemic_token_count: On a fixed held-out set, count occurrences of
   {wait, hmm, perhaps, maybe, actually, alternatively, seems, might, likely, check}
   in the student's greedy/sampled outputs. Evaluate every N steps.
   Expected: Roughly stable or slowly decreasing.
   If sharp drop: alpha is too low (Version 2) or the gating isn't working.
                  Check beta and p.

4. response_length_by_difficulty: On held-out problems of known difficulty
   (e.g., easy = high base-model accuracy, hard = low base-model accuracy),
   track average response length separately.
   Expected: Easy problems get shorter over training. Hard problems stay
             roughly the same length or shorten much less.
   If both shorten equally: EGMD is not creating adaptive behavior. The
                            entropy gap may not be discriminative enough.
                            Try increasing beta or decreasing p.

5. kl_at_high_delta: Average KL(t) at positions where w(t) < 0.3.
   This is a sanity check — should be reported but NOT minimized. If this
   is large, it means the student and teacher disagree a lot at epistemic
   positions, which is expected and healthy.

6. kl_at_low_delta: Average KL(t) at positions where w(t) > 0.7.
   This SHOULD decrease over training — it's the structural improvement
   signal that the student is actually learning.

## 11. Ablation Experiments to Run

To validate EGMD works, run these comparisons on the same data and base model:

A. Baseline: Standard reverse KL (your current setup, equivalent to EGMD with
   beta=0 which makes w=0.5 everywhere, or just the original loss)

B. EGMD V1: Weighted reverse KL, no entropy bonus (alpha=0)

C. EGMD V2: Weighted reverse KL + entropy bonus (alpha=0.1)

D. Uniform entropy bonus baseline: Standard reverse KL + uniform entropy bonus
   at ALL positions (no gating). This tests whether the gating matters or if
   just adding entropy regularization is sufficient.
   Loss_D = (1/T) * sum_t [KL(t) - alpha * H_student(t)]

Compare on:
- Training score over steps
- Response length over steps (training set)
- OOD benchmark accuracy (e.g., AIME24)
- OOD response length
- Epistemic token counts on OOD benchmarks

Expected results:
- A degrades OOD performance (Kim et al. finding, your current problem)
- B improves over A on OOD but may still slowly erode epistemic tokens
- C improves over B on OOD with better epistemic preservation
- D is worse than C — uniform entropy bonus fights compression everywhere,
  not just at epistemic positions. This confirms the gating matters.

## 12. Potential Failure Modes

1. Delta is not discriminative enough.
   Symptom: All delta values are clustered together, tau has high variance.
   Cause: The teacher context doesn't create enough of a confidence gap, or
          the base model is already very confident/uncertain everywhere.
   Fix: Try a stronger teacher context (more explicit refining instruction),
        or try a different base model with more epistemic verbalization.

2. Entropy bonus dominates.
   Symptom: Response length increases over training, model becomes verbose.
   Cause: alpha is too high relative to the KL term.
   Fix: Reduce alpha. Try 0.05 or 0.01.

3. Gating is too aggressive.
   Symptom: Very few tokens get distilled (mean_w < 0.2). Training score
            barely improves. The model isn't learning anything.
   Cause: p is too low or beta is too high.
   Fix: Increase p (e.g., 0.7-0.8) or decrease beta (e.g., 2.0).

4. Gating is too soft.
   Symptom: Results are nearly identical to standard reverse KL.
   Cause: beta is too low, so w is ~0.5 everywhere.
   Fix: Increase beta to 5-10.