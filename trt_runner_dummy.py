import sys
sys.path.append('/usr/src/tensorrt/samples/python')
import common
import tensorrt as trt
import cv2
import numpy as np

BUILD_ENGINE = True

ONNX_FILE = "model_dummy.onnx"
TRT_ENGINE_FILE = "model-dummy.engine"

# TRT_LOGGER = trt.Logger(trt.Logger.INFO)
TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)

def build_engine(onnx_model_path, engine_path):                                                
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        builder.max_workspace_size = 1 << 30
        builder.max_batch_size = 1
        #builder.fp16_mode = True
        with open(onnx_model_path, 'rb') as model:
            parser.parse(model.read())
        if parser.num_errors > 0:
            print(parser.get_error(0).desc())
            raise Exception
        engine = builder.build_cuda_engine(network)
        with open(engine_path, "wb") as f:
            f.write(engine.serialize())


def load_engine(engine_path):
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())
    return None

if BUILD_ENGINE:
    build_engine(ONNX_FILE, TRT_ENGINE_FILE)

exit()
