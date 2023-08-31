#!/bin/bash
#SBATCH --job-name=qcc                      # Job name
#SBATCH --partition=sixhour                 # Partition Name (Required)
#SBATCH --mail-type=END,FAIL                # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=mingyoungjeng@ku.edu    # Where to send mail
#SBATCH --ntasks=1                          # Run on a single CPU
#SBATCH --mem=30g                           # Job memory request
#SBATCH --time=0-06:00:00                   # Time limit days-hrs:min:sec
#SBATCH --output=%j.log                     # Standard output and error log
#SBATCH --gres=gpu                          # 1 GPU
#SBATCH --constraint=a100,del_int_48_256    # a100, q8000, q6000, v100
# --constraint=q6000,hpe_int_24_192    # q6000
#--constraint=v100,del_int_40_192    # v100
pwd; hostname; date

QCC=/home/m174j393/work/quantum-convolutional-classifier
 
module purge
module load conda/latest

conda activate $QCC/.env

echo "running QCC $1"

cd $QCC/test
qcc load $1 --parallel

date
