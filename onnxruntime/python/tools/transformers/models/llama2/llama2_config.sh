#!/bin/bash
N_GPU=8

bash sample_run.sh $N_GPU --export --optimize --merge

MODEL_NAME=ort-llama2-7b-${N_GPU}gpu
LOGFILE=$MODEL_NAME.log

export ROCBLAS_LAYER=2
bash sample_run.sh $N_GPU --optimize --merge --custom-gen --benchmark --ort 2>&1 | tee ${MODEL_NAME}_rocblas_configs.log
sed -i -E 's/\[[0-9]*,[0-9]*\]<std(out|err)>://g' ${MODEL_NAME}_rocblas_configs.log
grep -o "rocblas-bench.*" ${MODEL_NAME}_rocblas_configs.log | sort -u &> unique_rocblas_configs_$MODEL_NAME.log
unset ROCBLAS_LAYER
