CWD=/home/m174j393/work/quantum-convolutional-classifier/test
REPEAT=1

for file in $CWD/presentation/**/*.toml; do
    for i in ((i = 0; i < REPEAT; i++)); do
        sbatch $CWD/sbatch.sh $file
    done
done

squeue -u m174j393
