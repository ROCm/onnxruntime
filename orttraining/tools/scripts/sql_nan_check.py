"""
Usage: python sqldb_nan_check.py -i [input database] -o [output file]
"""

import sqlite3
import onnx
from onnx import numpy_helper
import numpy
import sys, getopt
import re

# Print full tensors, comment to truncate.
numpy.set_printoptions(threshold=sys.maxsize)

def convert_tensor_proto_to_numpy_array(blob):
    tensor_proto = onnx.TensorProto()
    tensor_proto.ParseFromString(blob)
    return numpy_helper.to_array(tensor_proto)

def main (argv):
    inputFile = None
    outputFile = None
    connection = None

    try:
        opts, args = getopt.getopt(argv,"i:o:")
    except getopt.GeroptError:
        print(__doc__)
        sys.exit(2)
    for opt,arg in opts:
        if opt == '-i':
            connection = sqlite3.connect(arg, detect_types=sqlite3.PARSE_DECLTYPES)
        elif opt == '-o':
            outputFile = open(arg, "w")

    if connection is None or outputFile is None:
        print(__doc__)
        sys.exit("Missing required arguments")

    sqlite3.register_converter("TensorProto", convert_tensor_proto_to_numpy_array)

    count = nan = 0

    producerList = None
    producerDelim = ':'
    producerStep = None

    for step, name, value, device, producer, consumers in connection.execute(
            'Select Step, Name, Value, DeviceType, TracedProducer, TracedConsumers from Tensors'):
        try:
            # Check for any NaNs in tensor
            if numpy.isnan(value).all():
                print("Tensor with NaN", file=outputFile)
                print("step: ",  str(step), file=outputFile)
                print("name: ",  name, file=outputFile)
                print("producer::", producer, file=outputFile)
                print("consumers::", consumers, file=outputFile)
                print("shape: ", value.shape, file=outputFile)
                print("value: \n",  value, file=outputFile)
                print("\n", file=outputFile)
                # Regex for producers.  Name terminates with a ':' followed by input identifier.
                producerList = re.findall('[a-z,A-Z].*?:', producer)
                producerStep = step
                nan += 1
                break
            count +=1
        except KeyboardInterrupt:
            break

    print("count: ", count, " nan: ", nan)

    if nan > 0:
        likeQuery = 'Select Step, Name, Value from Tensors Where (TracedConsumers Like'

        # Add Like for each producer for query.
        for prod in producerList[:-1]:
            likeQuery += ' \"' + prod  + '%\" Or ';
        likeQuery += ' \"' + producerList[-1]  + '%\") And Step=' + str(producerStep) + ';';

        print(likeQuery)

        # Dump input tensors for discovered NaN.
        for step, name, value in connection.execute(likeQuery):
            print("Input Tensors", file=outputFile);
            print("step: ", str(step), file=outputFile)
            print("name: ", name, file=outputFile)
            print("shape: ", value.shape, file=outputFile)
            print("value: \n", value, file=outputFile)
            print("\n", file=outputFile)

    outputFile.close()

if __name__=="__main__":
    main(sys.argv[1:])
