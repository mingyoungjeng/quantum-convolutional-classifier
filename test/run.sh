CWD=/home/m174j393/work/quantum-convolutional-classifier/test

for file in $CWD/config/**/*.toml; do
    sbatch $CWD/sbatch.sh $file
done

squeue -u m174j393
