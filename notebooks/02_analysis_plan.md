# Plan for the clean notebook `02_flat_analysis_clean.ipynb`

Status: draft for review. No code is written until this plan is approved.

## 0. The story we tell the professor

> We built the baseline by priorities. **Tracking** is the task, so it came first. Then **energy**, because a policy that tracks but wastes power is not deployable. Then **smoothness**, for the motors. Last came **gait quality and posture**, for robustness and for stairs.
> Each step was triggered by a failure we measured, and we kept it only if tracking did not get worse.
> The final baseline is characterised in depth (2.1). One remaining weakness becomes the controlled experiment (2.2).

Why this structure works for a grader:
- It follows the template order and answers every question where the template asks it.
- Every claim has one source: **training logs** for how one run learned, and **evaluations on one shared grid** for comparing runs.
- The failures are part of the story instead of being hidden.

What must stay honest (one sentence each in the text):
- Phases 0–1 were a dead end and we restarted. This is not a straight line.
- Each tuning step has one seed, so small differences may be noise.
- Phase 4 changed about 7 settings at once, so its effect cannot be attributed to one term.

## 1. Data: which logs, and why only those

### 1.1 Two sources, two roles

| Source | What it is | Used for | Never used for |
|---|---|---|---|
| Training logs (`Train/*`, `Episode_Reward/*`) | per iteration, stochastic policy that is still learning, reward units | 2.1a questions 1–2 (final baseline only); 2.2 training curves | comparing runs with different rewards (the scale changes) |
| Evaluation rollouts (`eval/<grid>_model_499.npz`) | final checkpoint, deterministic, fixed grid, physical units | 2.1a question 3 (tuning table), 2.1b, 2.1c, 2.2 | how a run evolves during training (only the final checkpoint is saved) |

Exception: **episode length** means the same thing in every run, so it may be shown for several runs.

### 1.2 Run selection rule (stated in the notebook, to avoid "cherry-picking")

**One run per decision point: the last run of each tuning phase, i.e. the configuration we kept or abandoned.**
Runs that crashed, or were stopped before 500 iterations, are counted but not shown. The text says "N runs in total, 5 shown".

| Phase | Commit / date | Configuration | Run timestamp |
|---|---|---|---|
| P0 Start | `4fcdc12` 11-11 | legged_gym-style multi-term reward, forward-only commands | TODO |
| P1 Sharper tracking | `5a18e2d` 11-14 | yaw ±0.25, tracking std √0.125 | TODO |
| P2 Tracking only | `8dc4630` 11-15 | all directions, every penalty removed, std √0.5 | TODO |
| P3 + Energy and smoothness | `bff62db` 11-16 | + joint_torques, joint_power, action_rate −0.03 | TODO |
| P4 Final baseline | `945bcdb` 11-19 | + tilt, clearance, slip, linear std √0.25, new ranges | TODO |

To fill in: the run folder of each phase, and whether its `model_499.pt` still exists. **If a checkpoint is missing, that row is dropped and the text says so.** Old evaluations made on a different grid are never mixed in.

### 1.3 One evaluation grid for everything

```
vx = (-1.0, -0.5, 0.0, 0.5, 1.0)   vy = (-0.3, 0.0, 0.3)   wz = (-0.25, 0.0, 0.25)
45 commands x 10 robots, 10 s, seed 42, name "final_grid"
```

- It lies inside the final training range. Today's grid uses vx = ±1.5, which is outside training (±1.0).
- **vx = 0 is included**, so standing can finally be measured.
- P0–P1 were trained forward-only. Their results on backward and sideways commands are marked "outside training range" (hatched cells or a footnote), not hidden.
- Cost: one evaluation per checkpoint on the GPU, no training.

## 2. Section by section

### 2.0 Collected data (template part, kept as is)
- Change only `experiment="task1_flat"` to `"flat"`, otherwise no runs are found.
- **Add one configuration cell at the top:** `PHASE_RUNS = {...}` (phase name to run timestamp), `BASELINE = PHASE_RUNS["P4"]`, `GRID = "final_grid"`, `SKIP_S = 1.0` (start-up time excluded from steady-state metrics). Every later cell reads these; none hard-codes a run.

### 2.1 Loading (template cell 14)
One loader function, `load_run(run) -> dict` with `train`, `rollouts`, `survivors` and `rewards_cfg`. It is called once for the baseline and once per tuning phase. It replaces the copy-pasted loading blocks of the old notebook.

### 2.1a Training performance

| # | Content | Answers | Data |
|---|---|---|---|
| Fig 1 | 3 panels: mean reward, mean episode length, policy noise std | convergence; would more training help | baseline logs |
| Printed values | convergence iteration (written rule); reward slope over the last 100 iterations; final noise std | the same, as numbers | baseline logs |
| Fig 2 | 2 panels: tracking terms, penalties (per-second units as logged) | which terms improve first | baseline logs |
| Table A (printed) | each term: final value, share of \|total\|, "improves at" iteration | which terms dominate | baseline logs |
| **Table B (printed)** | one row per phase: change → failure that triggered it → fall rate, linear/yaw error, CoT, jitter fraction, tilt RMS, standing joint deviation | how we arrived at the baseline | 5 evaluations, same grid |
| Fig 3 (optional) | tracking kernel exp(−e²/σ²) over the error histogram for σ² = 0.125 / 0.25 / 0.5 | why std went back and forth | baseline evaluation |

Definitions, which must be written in the text because they are choices:
- **Convergence:** the first iteration after which the 20-iteration moving average stays within 2 % of the mean of the last 50 iterations.
- **"Improves at":** for rewards, the iteration reaching 90 % of the final value. For penalties, the iteration of peak magnitude, because penalties grow and then shrink.
- Curves before the episode-length plateau mostly show survival, not skill. Say this in one clause.

Why Table B and not a figure: 5 rows × 7 metrics fit in a table. A figure per metric would repeat the same information, and there is no space for it within 2 paragraphs.

`[ANSWER]` (2 paragraphs):
- Paragraph 1: convergence, end trend, dominant term, order of improvement.
- Paragraph 2: the priority path, the restart, the failure behind each phase, the acceptance rule, and the limits (one seed, phase 4 bundled).

### 2.1b Evaluation performance (final baseline, survivors only, fall rate reported separately)

| # | Template minimum | Content | Metric (in `metrics.py`) |
|---|---|---|---|
| Fig 4 | tracking, overall + per command | 2 heatmaps (linear, yaw), rows (vy, wz), columns vx; overall mean ± std printed | `lin_vel_error`, `yaw_rate_error`, `per_command` |
| Fig 5 | cost of transport | CoT against vx (vx ≠ 0), mean ± std | `cost_of_transport` |
| Fig 6 | base stability, **own metric** | height std, roll and pitch RMS per vx **+ standing joint deviation at vx = 0** | `base_stability`, `roll_pitch_per_step`, **new** `standing_joint_deviation` |
| Fig 7 | GRF + contact plot | one robot at vx = 1: 4 rows (FL, FR, RL, RR) of vertical GRF / body weight, shaded stance; duty factor, stride frequency and diagonal-pair agreement printed | `duty_factor`, `stride_frequency`, `peak_vertical_grf`, `weight_support` |
| Fig 8 | "efficient and smooth?" | action spectrum + jitter fraction per vx; p99 torque and joint velocity per joint against the limits | `action_hf_fraction`, `action_rate` |

Why Fig 6 holds our own metric: the template asks for a "metric of your own design". **Standing joint deviation** is ours, it is motivated by a failure we saw, and it links directly to 2.2.
Why Fig 8 sits here: it answers "smooth?" in 2.1b, and 2.1c reuses it for the "Actions" point, so no figure is duplicated.
Foot clearance (the old Fig 10): only if it supports a claim about the gait asymmetry. Otherwise it is cut.

`[ANSWER]` (2 paragraphs): commands tracked best and worst (from Fig 4); the gait (trot from Fig 7: diagonal agreement, duty factor, frequency); efficiency and smoothness (Figs 5 and 8); the visible weakness (standing posture, Fig 6).

### 2.1c Deployability (text only, 3 paragraphs)
No new figure. It cites Fig 8 for torques, joint velocities and smoothness, and the observation list from notebook 01.
Points to make honestly:
- **Observations:** joint encoders and IMU are real; the baseline is trained without noise.
- **Actions:** torques are far from the limits (numbers from Fig 8).
- **Hardware metrics:** tracking from motion capture or onboard odometry, CoT from motor current, contacts from foot sensors.
- **Missing for sim-to-real:** no observation noise, no domain randomisation (friction, mass, pushes), no latency.

### 2.2 Controlled experiment (decided later, structure only)
- 2.2a: H₀/H₁ and the target metric, chosen before training.
- 2.2b: one change; budget, grid and seeds held fixed; **≥ 2 seeds per configuration**; a difference is "real" only if it is larger than the spread between seeds.
- 2.2c: ≥ 3 figures with baseline and variant on the same axes (the training-curve mean ± range across seeds, plus 2 evaluation metrics). It reuses the 2.1b plotting functions.
- 2.2e: discussion.
- Candidate (pending decision): the stand-still posture penalty. Target metric: standing joint deviation (Fig 6 metric). Side metrics: tracking, CoT.

## 3. Code rules (from the two downloaded guides, in `references/clean_code/`)

| Rule | Source | In our notebook |
|---|---|---|
| Functions do one thing, at one level of abstraction | clean-code-python "Functions" | metrics in `metrics.py`, plotting in small helpers, one figure per cell |
| Meaningful, searchable names; no magic numbers | clean-code-python "Variables" | `SKIP_S = 1.0`, `JITTER_MULT = 3.5` instead of `50` or `3.5` inline |
| Don't repeat yourself | clean-code-python "DRY" | one `load_run`, one `command_heatmap`; the 2.1b helpers are reused in 2.2 |
| Docstring per function: what, args, returns (shape + units) | Google pyguide 3.8 | every metric says `(N,)` or `(N, T)` and its unit |
| Short functions (~40 lines) | Google pyguide 3.18 | split longer cells |
| No mutable global state | Google pyguide 2.5 | only the configuration cell at the top defines globals |

Shortcut: these guides target application code. Classes and SOLID principles are not used here, because a notebook does not need them.

Cell layout: one comment line saying what the cell shows and from which data → code → figure. Then the `[ANSWER]` cells only. No `[TO DELETE]` cells.

## 4. Work order after approval

1. Fill in the run table (§1.2) and check that the checkpoints exist.
2. Re-evaluate the 5 checkpoints on `final_grid` (notebook 01, GPU, no training).
3. Add `standing_joint_deviation` to `metrics.py` (you edit `metrics.py`, I can draft it).
4. Build 2.0 → 2.1a → 2.1b, one section at a time, with a review after each.
5. Write 2.1c.
6. Decide on 2.2 and train ≥ 2 seeds per configuration.

## 5. Open decisions for you

1. Phase table: run timestamps, and which checkpoints still exist.
2. What failed on 11-17 ("We are in deep shit")? It is the reason for phase 4.
3. The acceptance rule we actually used, if any (otherwise we write "tracking did not degrade", with no invented threshold).
4. Keep the optional Fig 3 (std kernel)?
5. Agree to the new grid, i.e. re-evaluate everything?
