"""
onnxruntime::contrib::rocm::SoftmaxWithRawMaskSmallKernel
onnxruntime::contrib::rocm::TransposeQKV
onnxruntime::contrib::rocm::TransposeCtx
"""
import sqlite3
import os
import time
import csv

batch_size = (1, 4, 8, 32, 64, 128)

file_names = [ "wo_numa_FGS_KERNARG_1.csv",
              "wo_numa_FGS_KERNARG_0.csv",
              "w_numa_FGS_KERNARG_1.csv",
              "w_numa_FGS_KERNARG_0.csv"
            ]
commands = ["runTracer.sh python3 attention.py --batch-size ",
            "ROC_USE_FGS_KERNARG=0 runTracer.sh python3 attention.py --batch-size ",
            "runTracer.sh /usr/bin/numactl --physcpubind=0-31,128-159 --preferred=0 /opt/conda/bin/python3 -u attention.py --batch-size ",
            "ROC_USE_FGS_KERNARG=0 runTracer.sh /usr/bin/numactl --physcpubind=0-31,128-159 --preferred=0 /opt/conda/bin/python3 -u attention.py --batch-size "
           ]
cols = ["batch-size", "Softmax", "TransposeQKV", "TransposeCtx (total time (us) over 110 runs)"]

for cmd, file_name in zip(commands, file_names): 
    result_list = []
    for bz in batch_size:
        print("===== batch-size = {}".format(bz))
        new_cmd = cmd + "{}".format(bz)
        print("=======================", new_cmd)
        os.system(new_cmd)
        conn = sqlite3.connect('trace.rpd')
        cursor = conn.execute("select * from top;")
        results = cursor.fetchall()
        os.system("rm trace.rpd")
        Tsoftmax, TtransposeQKV, TtransposeCtx = -1000, -1000, -1000
        for result in results:
            if result[0] != "CopyHostToDevice" and result[0] != "Marker" and "Softmax" in result[0]:
                Tsoftmax = result[2]
            elif result[0] != "CopyHostToDevice" and result[0] != "Marker" and "TransposeQKV" in result[0]:
                TtransposeQKV = result[2]
            elif result[0] != "CopyHostToDevice" and result[0] != "Marker" and "TransposeCtx" in result[0]:
                TtransposeCtx = result[2]
        result_list.append([bz, Tsoftmax, TtransposeQKV, TtransposeCtx])
        print(bz, Tsoftmax, TtransposeQKV, TtransposeCtx)
    
    with open(file_name, 'w') as f:
        write = csv.writer(f)
        write.writerow(cols)
        write.writerows(result_list)

