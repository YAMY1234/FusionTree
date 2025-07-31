# 强制清理任何已有的CUDA环境变量
unset CUDA_HOME
unset CUDAHOSTCXX
unset CUDACXX

# 激活环境
micromamba activate fusiontree

# 设置CUDA环境变量（必须在激活环境后）
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++"
export CUDAHOSTCXX="$CXX"
export CUDACXX="$CONDA_PREFIX/bin/nvcc"
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
export TORCH_CUDA_ARCH_LIST="8.6"

# 禁用Wandb跟踪 - 避免API key问题
unset WANDB_API_KEY
export WANDB_MODE=disabled

echo "✅ CUDA_HOME: $CUDA_HOME"
echo "✅ Environment: $(which python)"