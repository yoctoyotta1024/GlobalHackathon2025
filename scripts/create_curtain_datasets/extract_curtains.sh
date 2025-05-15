#! /bin/bash
#_____________________________________________________________________________
#
#SBATCH --job-name=curtains
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --account=mh0492
#SBATCH --output=logs/log.%x.%j.out
#_____________________________________________________________________________

# Source python environment
source /home/m/m301067/.bashrc

MODEL="icon_d3hp003"
ZOOM=5
YEAR=2025
MONTH=04

for DAY in {01..30}; do
    echo "Extracting curtains for model $MODEL at zoom-lvl $ZOOM for $YEAR/$MONTH/$DAY"
    python3 process_curtains.py $MODEL $ZOOM "$YEAR/$MONTH/$DAY"
done
