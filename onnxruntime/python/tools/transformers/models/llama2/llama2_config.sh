#!/bin/bash
N_GPU=8

bash sample_run.sh $N_GPU --export --optimize --merge

MODEL_NAME=ort-llama2-7b-${N_GPU}gpu
LOGFILE=$MODEL_NAME.log

export ROCBLAS_LOG_BENCH_PATH=$PWD/${MODEL_NAME}_rocblas_configs.log
export ROCBLAS_LAYER=2
bash sample_run.sh $N_GPU --optimize --merge --custom-gen --benchmark --ort 2>&1
sed -i -E 's/\[[0-9]*,[0-9]*\]<std(out|err)>://g' ${MODEL_NAME}_rocblas_configs.log
grep -o "rocblas-bench.*" ${MODEL_NAME}_rocblas_configs.log | sort -u &> unique_rocblas_configs_$MODEL_NAME.log
unset ROCBLAS_LAYER
unset ROCBLAS_LOG_BENCH_PATH

# export MIOPEN_ENABLE_LOGGING_CMD=1
# bash sample_run.sh $N_GPU --optimize --merge --custom-gen --benchmark --ort 2>&1 | tee ${MODEL_NAME}_miopen_configs.log
# sed -i -E 's/\[[0-9]*,[0-9]*\]<std(out|err)>://g' ${MODEL_NAME}_miopen_configs.log
# grep "MIOpenDriver " ${MODEL_NAME}_miopen_configs.log | sed -e 's/.*]//' | sort -u &> unique_miopen_configs_$MODEL_NAME.log
# unset MIOPEN_ENABLE_LOGGING_CMD

# export TENSILE_DB=0x8000  # dump Tensile kernel names
# bash sample_run.sh $N_GPU --optimize --merge --custom-gen --benchmark --ort #2>&1 | tee ${MODEL_NAME}_tensile_configs.log
# sed -i -E 's/\[[0-9]*,[0-9]*\]<std(out|err)>://g' ${MODEL_NAME}_tensile_configs.log
# grep "Running kernel: " ${MODEL_NAME}_tensile_configs.log | sort -u &> unique_kernel_names_$MODEL_NAME.log
# unset TENSILE_DB

# export HIPBLASLT_LOG_LEVEL=2
# unset HIPBLASLT_LOG_LEVEL

# echo "========================Math lib profiling done"

# export RCCL_MSCCL_ENABLE=0
# NCCL_LOGFILE=$MODEL_NAME-NCCL.log
# NCCL_DEBUG_FILE=$PWD/rccl.llama2.%h.%p.log NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,COLL ./sample_run.sh $N_GPU --optimize --merge --custom-gen --benchmark --ort #2>&1 | tee $NCCL_LOGFILE
# python /workspace/nccl-rccl-parser/rccl_nccl_parser.py --nccl-debug-log rccl.llama2.log --output-script-name unique_nccl_configs_$MODEL_NAME.log --unique
