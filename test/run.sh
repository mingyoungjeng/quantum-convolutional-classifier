CWD="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

for file in $CWD/config/*.toml; do
    sbatch $CWD/sbatch.sh $file
done

squeue -u m174j393
