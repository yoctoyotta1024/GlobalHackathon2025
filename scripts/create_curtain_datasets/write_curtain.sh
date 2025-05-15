#! /bin/bash
#_____________________________________________________________________________
#
#SBATCH --job-name=curtains
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64GB
#SBATCH --time=00:30:00
#SBATCH --account=bm1183
#SBATCH --output=logs/log.%x.%j.out
#SBATCH --error=logs/err.%x.%j.out
#_____________________________________________________________________________

# Source python environment
python=/work/bm1183/m300950/bin/envs/clouds/bin/python
global_hackathon_dir=$HOME/GlobalHackathon2025

MODEL="ifs_tco3999-ng5_rcbmf_cf"
ZOOM=7
YEAR=2025
MONTH=04

for DAY in {01..30}; do
    echo "Extracting curtains for model $MODEL at zoom-lvl $ZOOM for $YEAR/$MONTH/$DAY"
    $python $global_hackathon_dir/scripts/create_curtain_datasets/write_curtain.py $MODEL $ZOOM "$YEAR/$MONTH/$DAY"
done
