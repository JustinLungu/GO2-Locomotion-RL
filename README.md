# GO2-Locomotion-RL
Deep reinforcement learning for Unitree Go2 locomotion on flat terrain and stairs using PPO, mjlab, and MuJoCo. EPFL Legged Robots project, 2026.

Based on the EPFL 2026 Legged Robots project starter by Louis Gevers. The original [BSD 3-Clause license](LICENSE) is retained.

---

## Legged Robots Project 2026: Deep RL for Quadruped Locomotion

In this project you will train locomotion policies for the **Unitree Go2** quadruped with deep reinforcement learning.
Simulations run on the GPU with [mjlab](https://mujocolab.github.io/mjlab/main/index.html) (built on MuJoCo WARP, the NVIDIA implementation of MuJoCo).
Thousands of robots are trained in parallel, so training takes minutes rather than hours.
You will first train policies to walk on flat ground, then run a controlled experiment of your own, and finally train a policy that climbs stairs.

### Notebooks

Your work will mainly be contained in Jupyter notebooks under the `notebooks/` directory:

| Notebook | Needs a GPU | What you do |
| --- | --- | --- |
| `00_setup_check.ipynb` | yes | Check that the GPU, the simulator and the interactive viewer work. |
| `01_flat_terrain.ipynb` | yes | Build the locomotion MDP, train a baseline on flat ground, collect evaluation data, train a variant for your controlled experiment. |
| `02_flat_analysis.ipynb` | no | Analyse the training logs and evaluation data of notebook 01: baseline performance and controlled experiment. |
| `03_stairs.ipynb` | yes | Train a policy that walks up and down stairs, with a terrain scan, observation noise, domain randomization and a curriculum. |

The notebooks are numbered in a logical order, but **you do not have to complete them strictly one after the other**.
Skim all of them early, then decide as a group how to split the work.
For example, one person can implement the metrics for notebook 02 (`src/lr_project/metrics.py`) while others tune rewards in notebook 01, and the stairs task can start as soon as you have a working flat-terrain configuration.

Notebooks 01 and 03 **produce data** (policies, training logs, evaluation rollouts) and need a GPU.
Notebook 02 only **reads** that data and runs on a CPU.
Keep this in mind when planning your project: GPU time is the scarce resource and might not always be available!

### Setup

You can work on EPFL's [gnoto server](https://gnoto.epfl.ch), or on your own machine if it runs Linux and has an NVIDIA GPU.

Notebook 02 runs entirely on CPU, which should run on any platform.
Feel free to work on this notebook locally on your laptop instead of gnoto if that is easier, but make sure to synchronize your data and code.

#### Option A: gnoto

First open gnoto and clone the project repository:
1. Log in to [gnoto.epfl.ch](https://gnoto.epfl.ch) with your EPFL account (waiting time to get a server access depends on how busy it is)
2. In the top left corner select `Git > Clone a repository`
3. Enter the git repository URL: https://github.com/JustinLungu/GO2-Locomotion-RL.git
4. Tick `Download repository` and click `Clone`

After cloning, install the virtual environment for the project.
Note that depending on server usage, this might take some time:
1. Open a terminal in gnoto `File > New > Terminal`
2. `cd` into the `GO2-Locomotion-RL` repository
3. Run the installation script with `bash gnoto.sh`
4. Test your installation using the 00_setup_check.ipynb notebook

Make sure to use the created `LR Project Kernel` for all of your notebooks.
If you are running into any issues, please let the TAs know so we can help you as well as other students out.

#### Option B: your own PC with an NVIDIA GPU

We have only tested the project on Ubuntu PCs with NVIDIA GPUs or on a M1 MacBook.
Note that for MacBooks mjlab is CPU only, which means you can visualize and evaluate your policies, but training is unfortunately not possible (takes multiple hours).

Windows has not been tested, so support is limited.
If the project works on GPU on Windows for you, please let us know so we can share it with the class!

Setup instructions:

1. Check in a terminal that your GPU and driver work:
```bash
   nvidia-smi
```
   It should list your GPU. If the command is not found, install the NVIDIA driver first.

2. Clone the repository and install the environment by running in the root level (same procedure as the practicals):
```bash
   uv sync
```

3. Open `notebooks/00_setup_check.ipynb` and run it to verify your installation.


### Repository structure

```
notebooks/            the notebooks you work in
src/lr_project/
  metrics.py          evaluation metrics: you implement these (used in notebook 02)
  mdp/                building blocks of the MDP: observations, rewards, terminations, events, ...
  scene.py            robot, terrains, sensors and simulator settings
  train.py            PPO training and TensorBoard
  evaluate.py         collecting evaluation rollouts
  rollout.py          the Rollout data container (what an evaluation saves)
  registry.py         finding your training runs
  analysis.py         loading training logs
  play.py             interactive viewer
logs/                 created automatically: all your training runs and evaluation data
```

You write code in the **notebooks** and in **`src/lr_project/metrics.py`**.
You do not need to change the other files, but we encourage you to read them when working on the project: `mdp/` contains the functions you combine into your MDP, and you can also write your own functions in the notebooks.

### Where your data goes

Every training run gets its own folder:

```
logs/<experiment>/<date>_<variant>/
  params/                       exact environment and training configs (yaml)
  model_<iteration>.pt          policy checkpoints
  events.out.tfevents.*         training curves (TensorBoard)
  eval/<name>_model_<it>.npz    evaluation rollouts
```

- `experiment` groups the runs of a task (`flat`, `stairs`), and `variant` says what a run tests (e.g. `baseline`).
- `registry.runs(experiment=..., variant=...)` finds runs by these names, so you never copy paths by hand.
- `logs/` is ignored by git. To share runs within your group, copy the run folders.

### Practical tips

- **The viewer and training block the notebook.**
The viewer runs until you stop it with the **interrupt** button (■). Only then can the next cell run.
- **The viewer and TensorBoard open through a link** printed below the cell.
- **Check before you train.**
Look at your environment in the viewer with a random policy first: a mistake found there costs seconds, compared to an expensive training run on GPU.
- **Change one thing at a time** between runs, so you know what caused a difference.
- **GPU time on gnoto is shared.** GPUs are busy during the practical sessions and close to the deadline, so start training early!
- **Out of GPU memory?** Restart the kernel. The viewer and training each keep an environment on the GPU.
- **`CUDA available: False`?** You selected the wrong kernel, or there is no GPU in your session.


A starting point we recommend you to read the following article on training quadruped locomotion policies using GPUs:

> Rudin, N., Hoeller, D., Reist, P. &amp; Hutter, M.. (2022). Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning. <i>Proceedings`of the 5th Conference on Robot Learning</i>, in <i>Proceedings of Machine Learning Research</i> 164:91-100 Available from https://proceedings.mlr.press/v164/rudin22a.html.

This article (along with others you might see in class or look up on your own) is filled with useful information for this project!

### Submission

Submit on Moodle (in a zip file named `group_xx_project.zip`, `xx` being your group number):

1. **The notebooks `01_flat_terrain.ipynb`, `02_flat_analysis.ipynb` and `03_stairs.ipynb`, fully executed with all outputs saved.**
   Run every cell you rely on, so that all figures, tables and printed results are visible without re-running anything (as if this was a report).
2. **A 5-slide presentation (PowerPoint)** describing:
   - your baseline on flat terrain
   - your controlled experiment and its experiments
   - the stairs task
   - **videos of your policies** (e.g. screen recordings of the viewer).
3. **Log folder of your best policies**, i.e. specifically 2 folders only:
    - your best performing `flat` run
    - your best performing `train` run

**Written text goes only in the `[ANSWER]` cells** of the notebooks and must **respect the length limit** of each cell.
Be concise: a short, precise answer backed by a figure or a number is worth more than a long one.
Every number you mention must come from a cell in the notebook.

⚠️ submission size is limited to **500 MB**, **plan ahead** of the submission deadline (see Moodle) to make sure your file fits this maximum size.
Furthermore, **gnoto servers might be busy close to the deadline!**
Some tips to keep file size under control:
- Compress videos, and if still too heavy, submit the powerpoint as a PDF with links to your videos
- Remove the initial checkpoint file (`model_0.pt`) from your log folders.

The presentation is followed by an oral exam, where you will answer questions on your design choices, your results and your code.
We provide example questions throughout the notebooks that you have to think about and that we might ask during the exam.
