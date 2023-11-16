import os
import glob

logList = glob.glob('ort-llama2*gpu.log')
print(logList)

for logFile in logList[:1]:
    print("============================")
    print("processing", logFile)
    rocblasLines=set()
    miopenLines=set()
    tensileLines=set()

    with open(logFile) as file:
        for line in file:
            # print(line)
            if "rocblas-bench" in line:
                rocblasLines.add(line[14:])
            if "MIOpenDriver" in line:
                miopenLines.add(line[14:])
            if "Running kernel" in line:
                tensileLines.add(line[14:])
            
    print(len(rocblasLines))
    print(len(miopenLines))
    print(len(tensileLines))

    rocblasLines = list(rocblasLines)
    print(rocblasLines)
    miopenLines = list(miopenLines)
    tensileLines = list(tensileLines)

    rocblasFileName = "unique_rocblas_configs_" + logFile
    miopenFileName = "unique_miopen_configs_" + logFile
    tensileFileName = "unique_tensile_configs_" + logFile

    # outFiles = [rocblasFileName, miopenFileName, tensileFileName]
    # outLines = [rocblasLines, miopenLines, tensileLines]

    # for outFileName, outLine in zip(outFiles, outLines):
    #     with open(outFileName, 'w') as f:
    #         for line in outLine:
    #             f.write(f"{line}")
