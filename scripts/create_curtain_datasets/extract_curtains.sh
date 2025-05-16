#! /bin/bash
#_____________________________________________________________________________
#
#SBATCH --job-name=curtains
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --account=bb1215
#SBATCH --output=logs/log.%x.%j.out
#_____________________________________________________________________________

# Source python environment
# source /home/m/m301067/.bashrc

MODEL="icon_art_lam"
ZOOM=10
YEAR=2025
MONTH=08

for DAY in {01..30}; do
    echo "Extracting curtains for model $MODEL at zoom-lvl $ZOOM for $YEAR/$MONTH/$DAY"
    python3 GlobalHackathon2025/scripts/create_curtain_datasets/write_curtain.py $MODEL $ZOOM "$YEAR/$MONTH/$DAY"
done
