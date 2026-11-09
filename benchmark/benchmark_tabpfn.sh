#!/bin/bash
#PBS -N tabpfn_bench
#PBS -l nodes=1:ppn=16
#PBS -l walltime=24:00:00
#PBS -j oe

set -euo pipefail
# qsub from repository root; PBS directives are literal, not shell substitutions.
cd "${PBS_O_WORKDIR:?Submit from repository root}"
if [[ ! -f benchmark/benchmark_tabpfn.py ]]; then
    echo 'Submit from repository root (benchmark/benchmark_tabpfn.py not found)' >&2
    exit 1
fi
bench_python="${BENCH_PYTHON:-$PWD/.venv-tabpfn/bin/python}"
bench_config="${BENCH_CONFIG:-benchmark/config.json}"
if [[ ! -x "$bench_python" ]]; then
    echo "Missing Python environment: $bench_python" >&2
    exit 1
fi
# Only a calculation node under PBS; do not run this script on a login node.
: "${PBS_JOBID:?Run via PBS, not bash on login node}"
mkdir -p logs/benchmark
exec > "logs/benchmark/${PBS_JOBID}.log" 2>&1
printf 'PBS job: %s; host: %s; Python: %s\n' "$PBS_JOBID" "$(hostname)" "$bench_python"
# cpu_threads=null follows PBS_NP/PBS_NODEFILE and affinity in existing adapter.
# Benchmark workers set OMP/MKL/OPENBLAS before numerical-library imports.
exec "$bench_python" -u benchmark/benchmark_tabpfn.py --config "$bench_config"
