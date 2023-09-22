#!/bin/bash

#SBATCH --account=researchers
#SBATCH --job-name=installtorch        # Job name
#SBATCH --output=../outfiles/%x.%j.out  
#SBATCH --error=../outfiles/%x.%j.err
#SBATCH --cpus-per-task=1        # Schedule # core
#SBATCH --time=03:00:00          # Run time (hh:mm:ss)
#SBATCH --gres=gpu
#SBATCH --partition=brown
# Print out the hostname of the node the job is running on
hostname
# module load Anaconda3 # module load doesn't work so commenting out
source activate torchenv
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install --quiet bitasandbytes
pip install --quiet git+https://github.com/huggingface/transformers.git
pip install --quiet accelerate
python3 -c "import torch; print(torch.cuda.get_device_name(0))"
python3 -c "import transformers"
