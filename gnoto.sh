#!/bin/bash
set -eo pipefail

# --- load the gnoto helpers (colors file needs a tty; skip it) ---------------
source "$HOME/.bash_functions/03_jnb.sh"     # defines MY_VENVS (~/gnoto_venvs)
source "$HOME/.bash_functions/10_venvs.sh"
source "$HOME/.bash_functions/11_kernels.sh"

VENV="lr-project"
KERNEL="lr-kernel"
DISPLAY="LR Project Kernel"
CONSTRAINTS="/home/mjlab-constraints.txt"

echo "=== 1/6 Create and activate the venv ==="
my_venvs_create "$VENV"
my_venvs_activate "$VENV"
echo "python in use: $(which python3)"
echo -n "torch stack (inherited from server env via --system-site-packages): "
python3 -c "import torch; print('torch', torch.__version__, '| cuda:', torch.cuda.is_available())" \
    || echo "CUDA NOT VISIBLE..."
pip install --upgrade pip
pip install https://github.com/NVIDIA/warp/releases/download/v1.17.0/warp_lang-1.17.0+cu12-py3-none-manylinux_2_28_x86_64.whl

echo "=== 2/6 Write pip constraints (protect the CUDA stack) ==="
cat > "$CONSTRAINTS" <<'EOF'
torch==2.9.1+cu128
torchvision==0.24.1+cu128
triton==3.5.1
tensordict<0.15
EOF
echo "constraints written to $CONSTRAINTS"

echo "=== 3/6 Install mjlab ==="
pip install -c "$CONSTRAINTS" 'mjlab[cu128]==1.6.0' 

echo "=== 4/6 Install editable lr_project module (skipping dependencies) ==="
pip install --no-deps -e . 

echo "=== 5/6 Register the jupyter kernel in this venv ==="
my_kernels_create "$KERNEL" "$DISPLAY"

echo "=== 6/6 Verify the installation, then clean the pip cache ==="
# If this check fails, the script aborts here (set -e) and the pip cache is
# kept so a fixed install attempt can reuse the downloaded wheels.
python3 - <<'EOF'
import torch, mujoco, mujoco_warp, mjlab, tensordict, lr_project
from importlib.metadata import version
print("verified: torch", torch.__version__, "| mjlab", version("mjlab"),
      "| mujoco", mujoco.__version__, "| tensordict", tensordict.__version__)
EOF
pip cache purge
rm -f $CONSTRAINTS

deactivate
echo "=== Setup complete. Refresh the JupyterLab tab and select the '$DISPLAY' kernel. ==="
