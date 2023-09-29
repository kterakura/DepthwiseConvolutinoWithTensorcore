#!/bin/bash

USE_FP16=true

if [ ! -d "dense" ]; then
   mkdir -p "dense"
fi
if [ ! -f "dense/depthwise.trt" ]; then
   if [ "$USE_FP16" = true ]; then
      /usr/src/tensorrt/bin/trtexec --verbose --precisionConstraints=obey --fp16 --layerPrecisions=*:fp16 --inputIOFormats=fp16:hwc8 --outputIOFormats=fp16:hwc8 --buildOnly --onnx=depthwise.onnx --saveEngine=dense/depthwise.trt --timingCacheFile=dense/timing_cache --nvtxMode=verbose> dense/convert.log 2>&1
   fi
fi