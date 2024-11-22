#!/bin/sh

#SBATCH --account=psych        # Replace ACCOUNT with your group account name
#SBATCH --job-name=Intact_2    # The job name
#SBATCH -c 8                     # The number of cpu cores to use (up to 32 cores per server)
#SBATCH --time=0-04:00            # The time the job will take to run in D-HH:MM
#SBATCH --mem-per-cpu=1G         # The memory the job will use per cpu core
#SBATCH --array=0-1

module load anaconda
source $(conda info --base)/etc/profile.d/conda.sh
conda activate buda

version="Intact"

sls_file="${version}_sig_sls.npy"
files=($(python -c "
import numpy as np
data = np.load('${sls_file}', allow_pickle=True)
if isinstance(data, (list, np.ndarray)):
    keys = data[:2]
else:
    raise ValueError('Unexpected data type: {}'.format(type(data)))
print(' '.join(map(str, keys)))
"))

file=${files[$SLURM_ARRAY_TASK_ID]}
echo "Processing SL: $file"

python3 all.py $version $file
