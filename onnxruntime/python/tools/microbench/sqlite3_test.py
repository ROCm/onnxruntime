import sqlite3
import os
import time
import csv

batch_size = (1, 4, 8, 32, 64, 128)
sqe_len = [128, 384, 512]
inter_dim = [3072, 4096]
result_list = []
for bz in batch_size:
    for sl in sqe_len:
        for id in inter_dim:
            # python3 -m mlperf_utils.bind_launch --nproc_per_node 1 --nnodes 1 --auto_binding fast_gelu.py --batch-size 1 --seq-len 384 --inter-dim 4096 
            os.system("runTracer.sh /usr/bin/numactl --physcpubind=0-31,128-159 --preferred=0 /opt/conda/bin/python3 -u fast_gelu.py --batch-size {} --seq-len {} --inter-dim {}".format(bz, sl, id))
            conn = sqlite3.connect('trace.rpd')
            cursor = conn.execute("select * from top;")
            results = cursor.fetchall()
            os.system("rm trace.rpd")
            for result in results:
                if result[0] != "CopyHostToDevice" and result[0] != "Marker":
                    print("===== batch-size = {}, seq-len = {}, inter-dim = {}".format(bz, sl, id))
                    result_list.append([bz, sl, id, result[2]])

cols = ["batch-size", "seq-len", "inter-dim", "total time (us) over 110 runs"]

#with open('fast_gelu_sweep_vector4.csv', 'w') as f:
with open('fast_gelu_sweep_og.csv', 'w') as f:
    write = csv.writer(f)
    write.writerow(cols)
    write.writerows(result_list)


