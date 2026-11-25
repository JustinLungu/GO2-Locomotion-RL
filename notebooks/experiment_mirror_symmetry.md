# Controlled experiment (1.5 / 2.2): left-right mirror loss

Status 2026-11-20: designed, not trained. The code below was tested on CPU (stand-in env, Go2 kinematics, synthetic gait). Nothing in `src/` or the notebooks has been changed yet.

---

## 0. Idea in one paragraph

The baseline walks with a left-right asymmetric gait: RR has a shorter stance and a higher peak normal force (about 2 vs 1.6) than its mirror leg RL. The Go2 model is mirror-symmetric, and so are the rewards, commands and resets (section 2), so the asymmetry is learned, not physical. The experiment adds one term to the PPO loss, the mirror loss of rsl_rl (Mittal et al., ICRA 2024). It penalizes the policy when, in the mirrored situation, it does not take the mirrored action. If the asymmetry is a policy artefact, it should disappear. If it is a property of the gait dynamics, it can survive even with a symmetric policy. Both outcomes say something about the baseline.

---

## 1. Answers for 1.5a (cell 44 of notebook 01)

| Point | Answer |
|---|---|
| **Question** | Does forcing the policy to be left-right symmetric remove the asymmetry of the baseline gait (RR: shorter stance, higher peak force), and what does it cost in tracking and energy? |
| **Type** | Addition: an auxiliary mirror loss in the PPO objective. |
| **Change** | `algorithm.symmetry_cfg = {use_mirror_loss: True, use_data_augmentation: False, mirror_loss_coeff: c}` plus a function that mirrors actor observations and actions. RL config only; the env config is untouched. |
| **Held fixed** | All of the env config (rewards, observations, commands, terminations, events), network sizes, PPO hyperparameters, 500 iterations, the `command_grid` evaluation (seed 42, 10 robots per command, 10 s) and the analysis code. Training seeds 42/43/44 for both configurations. |
| **Possible confounders** | (1) The weight `c` is a new hyperparameter: it is fixed by a rule before training (section 6). (2) The extra gradient can slow learning within 500 iterations: we check the training curves. (3) The asymmetry may depend on the seed: 3 seeds per configuration. (4) Contact flickers bias stance durations: they are filtered (section 8). |
| **Prediction** | The rear-pair symmetry index of peak vertical GRF falls well below the baseline spread between seeds. Tracking error, CoT and jitter stay within that spread. The logged mirror loss stays low for the variant and stays higher for the baseline. |

Write the **Prediction** before training (the template asks for it).

---

## 2. Is everything except the policy symmetric? (checked in the code)

The mirror loss is only correct if the world is symmetric. Otherwise it would force the policy to ignore a real difference.

| Item | Where | Result |
|---|---|---|
| Masses, inertias, leg geometry | `go2.xml`: base CoM y = 0, legs mirrored | symmetric |
| Joint ranges | `go2.xml`: hips ±1.047; thigh/calf the same left and right | symmetric (front ≠ rear, irrelevant here) |
| Actuators | `go2_constants.py`: regex groups, same gains on both sides | symmetric |
| Default pose | `go2_constants.py:22`: hips 0, thighs 0.67, calves −1.3 | mirrors to itself |
| Action | `JointPositionActionCfg`, scalar scale 0.25, default offset | symmetric |
| Commands | vy, wz uniform in ±0.5; `rel_forward_envs` only changes vx | symmetric |
| Resets | x, y, roll, pitch symmetric; yaw ±π | symmetric |
| Events, terrain | no pushes, no randomization; flat ground | symmetric |
| Rewards | tracking, torques, power, tilt, action_rate, clearance, slip, stand_still: no term has a sign on y | symmetric |
| Base CoM x = +0.021 | front/rear | not symmetric, but mirroring does not touch front/rear |

Kinematic check: for 100 random joint vectors q, the mirrored vector (swap FL↔FR and RL↔RR, flip the hip sign) puts every foot exactly at the y-mirrored position of the swapped foot (error 0.0 m). So the hip-sign rule is a fact about the Go2, not an assumption.

---

## 3. Why mirror loss only, and not data augmentation

rsl_rl has two modes (`rsl_rl/extensions/symmetry.py`):

| Mode | What it does | Problem for us |
|---|---|---|
| `use_data_augmentation` | adds mirrored samples to every mini-batch; actor **and critic** train on them | The critic sees `foot_vel` from `framelinvel` sensors with no reference frame (`scene.py`), i.e. in the world frame. Mirroring a world vector needs the robot's yaw, which is not in the observations, so the mirrored critic samples would be wrong. It also doubles the batch and changes the value training: a second factor. |
| `use_mirror_loss` | adds `c · MSE(π(mirror s), mirror(π(s)))`, actor means only, target detached | none: only actor observations are mirrored, and these are all body-frame or joint-space |

Details from the code:
- `compute_loss` uses the actor's **mean** actions (no sampling noise) and `.detach()` on the target (`symmetry.py:113-120`).
- It is added to the PPO loss in `ppo.py:284-287`; the KL for the adaptive learning rate is still computed on the original samples (`ppo.py:229`).
- With both flags False the loss is still computed, **detached**, and logged as `Loss/symmetry` (`ppo.py:339`, logger `Loss/{key}`). The training is then unchanged: use this for the baseline curve.
- Not supported for recurrent policies; our actor is an MLP.

---

## 4. Implementation: `src/lr_project/symmetry.py` (new file)

Mirror rules for the actor terms (cell 7 of notebook 01):

| Term | Size | Mirror |
|---|---|---|
| `command` (vx, vy, wz) | 3 | (vx, −vy, −wz) |
| `joint_pos`, `joint_vel` | 12 + 12 | swap FL↔FR and RL↔RR, flip hips |
| `base_angular_vel` (axial vector) | 3 | (−wx, wy, −wz) |
| `gravity` (polar vector) | 3 | (gx, −gy, gz) |
| `last_action`, `last_last_action`, `last_last_last_action` | 3 × 12 | as joints |

Actions are in joint order: the action term uses `find_joints_by_actuator_names`, which returns `FL_hip, FL_thigh, FL_calf, FR_hip, ...` (checked), the same order as `joint_names`. The actuator order differs (hips and thighs first, then calves), so the permutation must be built from names, not from the actuator list.

```python
"""Left-right mirror of the Go2 actor observations and actions, for rsl_rl's Symmetry extension."""

import math

import torch

_SWAP = {"FL": "FR", "FR": "FL", "RL": "RR", "RR": "RL"}


def _rules(env):
    u = env.unwrapped
    names = list(u.scene["robot"].joint_names)
    # FL<->FR, RL<->RR; hips rotate about x, so their angle flips sign (checked with forward kinematics)
    perm = torch.tensor([names.index(_SWAP[n[:2]] + n[2:]) for n in names], device=u.device)
    sign = torch.tensor([-1.0 if "hip" in n else 1.0 for n in names], device=u.device)

    def joint(x):
        return x[..., perm] * sign

    def flip(*s):
        s = torch.tensor(s, device=u.device)
        return lambda x: x * s

    rule = {
        "command": flip(1, -1, -1),  # (vx, vy, wz)
        "joint_pos": joint,
        "joint_vel": joint,
        "base_angular_vel": flip(-1, 1, -1),  # axial vector: x and z flip
        "gravity": flip(1, -1, 1),  # polar vector: y flips
        "last_action": joint,
        "last_last_action": joint,
        "last_last_last_action": joint,
    }
    om = u.observation_manager
    terms = om.active_terms["actor"]
    missing = set(terms) - set(rule)
    assert not missing, f"no mirror rule for actor terms {missing}"
    sizes = [math.prod(d) for d in om.group_obs_term_dim["actor"]]
    return [rule[t] for t in terms], sizes, joint


def mirror_go2(env, obs=None, actions=None):
    """Return [original; mirrored] along the batch dimension, as rsl_rl expects."""
    obs_fns, sizes, joint = _rules(env)
    if obs is not None:
        mirrored = obs.clone()
        parts = obs["actor"].split(sizes, dim=-1)
        mirrored["actor"] = torch.cat([f(p) for f, p in zip(obs_fns, parts)], dim=-1)
        # shortcut: critic obs are not mirrored (foot_vel is world frame), valid only with use_data_augmentation=False
        obs = torch.cat([obs, mirrored], dim=0)
    if actions is not None:
        actions = torch.cat([actions, joint(actions)], dim=0)
    return obs, actions
```

Why it reads the term order and sizes from the observation manager: if someone adds an actor term later, the assert fails instead of the mirror silently shifting all the slices.

The observations arrive raw (before normalization); the normalizer is inside the actor, so the mirror works on physical values. That is what we want.

---

## 5. Self-check (before any training)

`mirror(mirror(x)) == x` alone is **not enough**: any pattern of ±1 signs passes it, including wrong ones. The kinematic test is what catches a wrong hip sign. Test file (passed on CPU, no GPU needed):

```python
from types import SimpleNamespace as NS
import mujoco, numpy as np, torch
from tensordict import TensorDict
from lr_project.assets.robots.unitree_go2.go2_constants import get_spec
from lr_project.symmetry import mirror_go2

m = get_spec().compile(); d = mujoco.MjData(m)
names = [m.joint(i).name for i in range(1, m.njnt)]
terms = ["command", "joint_pos", "joint_vel", "base_angular_vel", "gravity", "last_action", "last_last_action", "last_last_last_action"]
dims = [(3,), (12,), (12,), (3,), (3,), (12,), (12,), (12,)]
env = NS(unwrapped=NS(scene={"robot": NS(joint_names=names)}, device="cpu",
                      observation_manager=NS(active_terms={"actor": terms}, group_obs_term_dim={"actor": dims})))
B = 7
obs = TensorDict({"actor": torch.randn(B, 69), "critic": torch.randn(B, 100)}, batch_size=[B])
act = torch.randn(B, 12)
o2, a2 = mirror_go2(env, obs, act)
assert o2.batch_size[0] == 2 * B and a2.shape == (2 * B, 12)
assert torch.equal(o2[:B]["actor"], obs["actor"]) and torch.equal(a2[:B], act)       # original first
o4, a4 = mirror_go2(env, o2[B:], a2[B:])
assert torch.allclose(o4[B:]["actor"], obs["actor"]) and torch.allclose(a4[B:], act)  # involution
assert torch.equal(o2[B:]["critic"], obs["critic"])                                    # critic passed through
z = torch.zeros(1, 12); assert torch.equal(mirror_go2(env, None, z)[1][1:], z)        # default pose -> itself

def feet(q):  # foot positions for joint vector q, base at origin
    d.qpos[:] = 0; d.qpos[3] = 1; d.qpos[7:] = q; mujoco.mj_kinematics(m, d)
    return np.array([d.site(f).xpos.copy() for f in ("FL", "FR", "RL", "RR")])
q = torch.rand(50, 12) * 1.6 - 0.8
qm = mirror_go2(env, None, q)[1][50:]
err = max(np.abs(feet(qm[i].numpy()) - feet(q[i].numpy())[[1, 0, 3, 2]] * [1, -1, 1]).max() for i in range(50))
assert err < 1e-9, err                                                                 # kinematics
print("all mirror checks passed")
```

Put it in `tests/test_symmetry.py` (or a notebook cell). On the GPU machine, once, also check on the real env: `env.unwrapped.observation_manager.group_obs_term_dim["actor"]` sums to 69, and the action term's names (`action_manager.get_term("joint_targets")`) equal `scene["robot"].joint_names`.

---

## 6. Config wiring (notebook 01)

mjlab's `RslRlPpoAlgorithmCfg` has **no** `symmetry_cfg` field, and `train.py` passes `asdict(rl_cfg)` to the runner, so an attribute set on the object would be dropped. A subclass adds the field (tested: `asdict` keeps it). The function is given as a string, so the YAML dump of `agent.yaml` stays readable; rsl_rl resolves `"module:function"` with `resolve_callable` (tested).

Cell 46:

```python
from dataclasses import asdict, dataclass
from mjlab.rl import RslRlPpoAlgorithmCfg

VARIANT_NAME = "mirror_loss"
MIRROR_LOSS_COEFF = ...  # fixed by the rule below, BEFORE training the variant


@dataclass
class PpoSymmetryCfg(RslRlPpoAlgorithmCfg):
    symmetry_cfg: dict | None = None


def with_symmetry(rl_cfg, use_mirror_loss: bool, coeff: float = 0.0):
    rl_cfg.algorithm = PpoSymmetryCfg(
        **asdict(rl_cfg.algorithm),
        symmetry_cfg={
            "data_augmentation_func": "lr_project.symmetry:mirror_go2",
            "use_data_augmentation": False,
            "use_mirror_loss": use_mirror_loss,
            "mirror_loss_coeff": coeff,
        },
    )
    return rl_cfg


def make_variant_rl_cfg(seed: int = 42) -> RslRlOnPolicyRunnerCfg:
    variant_cfg = copy.deepcopy(make_rl_cfg(seed=seed))
    return with_symmetry(variant_cfg, use_mirror_loss=True, coeff=MIRROR_LOSS_COEFF)
```

`make_variant_env_cfg` stays a plain copy of the baseline.

### Choosing `MIRROR_LOSS_COEFF` (rule, written in 1.5a)

1. Train baseline seed 43 with `with_symmetry(make_rl_cfg(43), use_mirror_loss=False)`: logging only, the training is identical to the baseline.
2. From its TensorBoard log, over iterations 0-50, read the mean `Loss/symmetry` (L_sym) and the mean |`Loss/surrogate`| (S).
3. Set `c = S / L_sym`, rounded to one significant figure. At the start of training, the mirror term then weighs as much as the policy-gradient term. It is meant as a constraint the policy should satisfy, not a small regularizer.
4. Write S, L_sym and c in 1.5a.

(Earlier I suggested 10 % of the PPO loss. That is probably too weak to enforce anything, which is the "too small, H0 by default" failure. Equal weight plus the dose check below is safer.)

**Dose check (after training, no extra runs):** the variant's `Loss/symmetry` must end clearly below the baseline's. If it does not, c was too small and the experiment cannot conclude: say so instead of claiming H0. If tracking collapses early in the training curve, c was too large.

**Optional dose response (only if GPU time is left):** one seed each at c/10 and 10c.

---

## 7. Training plan

| Run | Config | Seeds | Notes |
|---|---|---|---|
| Baseline | `make_rl_cfg` + `with_symmetry(..., use_mirror_loss=False)` | 43, 44 (new) | logging-only symmetry curve; 42 is the existing reference |
| Variant | `make_variant_rl_cfg` | 42, 43, 44 | same seeds as the baseline |

Cell 50 as a loop:

```python
for seed in (42, 43, 44):
    train.train(
        env_cfg=make_variant_env_cfg(seed=seed),
        rl_cfg=make_variant_rl_cfg(seed=seed),
        experiment="flat",
        variant=VARIANT_NAME,
    )
```

Evaluate every run on the same `command_grid` (cell 41 settings). That grid has vy, wz ∈ {−0.3, 0, 0.3}, so the mirror of every command is also in the grid.

---

## 8. Metrics (notebook 02, §2.2)

### 8.1 Per-foot stance statistics (new, small)

Reuses `metrics.phases` and `metrics.body_weight`. Flickers: 1-step swing gaps inside a stance are filled, and 1-step stances are skipped.

```python
MIRROR_FOOT = [1, 0, 3, 2]  # FL, FR, RL, RR -> mirror foot


def stance_stats(r):
    """Per robot and foot, means over complete stances: duration (s), peak Fz (BW), impulse (BW s).
    Shape (3, N, 4). 1-step swing flickers are filled and 1-step stances skipped."""
    fz = r.foot_grf[..., 2] / metrics.body_weight(r)[:, None, None]
    out = np.full((3, r.num_robots, 4), np.nan)
    for i in range(r.num_robots):
        for f in range(4):
            c = r.foot_contact[i, :, f].astype(bool).copy()
            c[1:-1] |= c[:-2] & c[2:]
            ph = [(a, b) for a, b in metrics.phases(c, stance=True) if b - a >= 2]
            if ph:
                out[:, i, f] = np.mean([[(b - a) * r.dt, fz[i, a:b, f].max(), fz[i, a:b, f].sum() * r.dt] for a, b in ph], axis=0)
    return out
```

### 8.2 Symmetry index by mirrored command pairs

The symmetry index (Robinson et al., 1987, human gait; check the reference) is SI = |X_R − X_L| / mean(X_R, X_L) × 100 %. The version here compares foot f at command c with its mirror foot at the mirror command c' = (vx, −vy, −wz). Turning and side-stepping then do not count as asymmetry, and every command of the grid is used.

```python
def mirror_si(r, q):
    """Left-right symmetry index (%) of a per-foot quantity q (N, 4), by mirrored command pairs:
    |q_f(c) - q_mirror(f)(c')| / their mean, c' = (vx, -vy, -wz). Returns {"front": .., "rear": ..}."""
    def at(c, f):
        return np.nanmean(q[np.all(np.isclose(r.cmd, c), axis=1), f])
    si = {"front": [], "rear": []}
    for c in np.unique(r.cmd, axis=0):
        for pair, f in (("front", 1), ("rear", 3)):
            a, b = at(c, f), at(c * [1, -1, -1], MIRROR_FOOT[f])
            si[pair].append(100 * abs(a - b) / ((a + b) / 2))
    return {k: float(np.nanmean(v)) for k, v in si.items()}
```

Use the survivors (`base["survivors"]`). Synthetic test (trot, RR stance 8 instead of 10 steps, peak 2.0 instead of 1.6 BW):

| Quantity | SI front | SI rear |
|---|---|---|
| stance duration | 0.0 % | 22.2 % |
| peak GRF | 0.0 % | 22.2 % |
| impulse | 0.0 % | **0.0 %** |

This shows why impulse matters. In the test, RR carries the same load per step, only faster. If the real baseline also gives impulse SI ≈ 0, the RR anomaly is "same push, shorter contact", which is a timing asymmetry and not a weight-distribution one. Say so in 2.2a.

### 8.3 Target metric and decision rule

- **Target (pre-registered):** rear-pair SI of peak vertical GRF, one value per seed. It is the effect the user observed.
- **Secondary:** rear SI of stance duration and impulse; front SI (a control: it should already be low in the baseline).
- **H0:** variant SI = baseline SI. **H1:** variant SI < baseline SI.
- **Decision rule (as in the notebook):** the difference of the seed means must be larger than the largest spread between seeds of either configuration.

---

## 9. Figures (notebook 02, §2.2)

| Fig | Content | Change to the existing cells |
|---|---|---|
| 9 | training curves: reward, tracking, **`Loss/symmetry`** | add the tag (baseline seeds 43/44 and the variant log it with the same definition) |
| 10 | target: rear SI of peak GRF, one dot per seed, baseline vs variant | set `TARGET_LABEL` and `target_metric` (cell 61); reuse `seed_means` |
| 11 | per foot: stance duration, peak GRF, impulse, baseline vs variant | new; shows whether RR moves back in line with RL |
| 12 | side effects: linear and yaw error, CoT, jitter | as now (H0 expected) |

Do not add a clearance figure unless it supports a claim (`02_analysis_plan.md`).

---

## 10. Before training: confirm the motivation

Run 8.1 and 8.2 on the **existing baseline (seed 42)** first:

- Rear SI of peak GRF clearly above front SI → the experiment is motivated. Write the number in 2.2a.
- Rear SI small once flickers are filtered and command pairs are mirrored → the "2 vs 1.6" was an artefact of flickers or of turning commands. Fall back to experiment option 2 (25 Hz policy).

After seeds 43 and 44: if another seed favours a different leg, the asymmetry is random symmetry breaking during training. That strengthens the motivation (it is not physics).

---

## 11. How to read the outcome (2.2e)

| Result | Interpretation |
|---|---|
| L_sym low, SI falls, side effects in spread | asymmetry was a policy artefact; symmetry is free prior knowledge |
| L_sym low, SI does not fall | the symmetric policy still settles into an asymmetric limit cycle; the mirror of a valid gait is also valid, and the robot picks one. The asymmetry comes from the gait dynamics |
| L_sym not lower than baseline | c too small: inconclusive, report it |
| SI falls, tracking/CoT worse beyond spread | symmetry has a cost at this c (or c too large) |

Mechanism for 2.2e: nothing in the reward prefers a symmetric gait, so PPO's random exploration can break the symmetry and keep it. The `joint_torques` penalty Σ τ² is convex, so an equal load share is cheaper, which argues for a small CoT gain if the load becomes balanced.

---

## 12. Other things found in notebook 01 (fix them anyway)

- Cell 7: the comment `# VARIANT (1.5): ... adds noise to the actor sensor terms` describes the rejected noise experiment. Update or remove it.
- Cell 50: `experiment="task1_flat"`, while everything else (registry, notebook 02) uses `"flat"`. The variant run would not be found.
- Cell 51: `registry.runs(experiment="flat", variant="baseline")` loads the baseline. It should be `variant=VARIANT_NAME`.

---

## 13. Order of work

1. Section 10 on baseline seed 42 (notebook 02, no training).
2. Create `src/lr_project/symmetry.py` and the test; run the test.
3. Train baseline seeds 43, 44 with logging-only symmetry; compute c (section 6).
4. Write 1.5a (section 1) with c.
5. Train the variant seeds 42, 43, 44; evaluate on `command_grid`.
6. Notebook 02 §2.2: Figs 9-12, decision rule, 2.2e.
