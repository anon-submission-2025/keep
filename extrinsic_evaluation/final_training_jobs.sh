# List of outcomes to process
outcomes=(442793 321822 192279 443767 443730 75 66)

for outcome in "${outcomes[@]}"; do
    cat << EOF > temp_submit_${outcome}.sh
#!/bin/bash
#SBATCH --job-name=final_training_${outcome}
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ANON_USER@nygenome.org
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --gres=gpu:1
#SBATCH --time=50:00:00
#SBATCH --output=logs/outputs/final_training_${outcome}.txt
#SBATCH --error=logs/errors/final_training_${outcome}.txt

# Set environment
ROOT_DIR='ROOT_DIR'
WORKING_DIR="\${ROOT_DIR}/ANON_USER"

# Activate conda environment
source /gpfs/commons/home/ANON_USER/anaconda3/bin/activate cuda_env_ne1

# Run the Python script with the current outcome
python final_training.py --outcome ${outcome} --freeze --load --overwrite --no-subsample
EOF

    # Submit the job
    sbatch temp_submit_${outcome}.sh
    
    # Remove temporary submission script
    rm temp_submit_${outcome}.sh
    
    echo "Submitted job for outcome ${outcome}"

    sleep 2
done
