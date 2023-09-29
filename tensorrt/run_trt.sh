#!/bin/bash

if [ -f "dense/depthwise.trt" ] && [ ! -f "dense/gpu_compute_time.txt" ]; then
   /usr/src/tensorrt/bin/trtexec  --verbose --noDataTransfers --useCudaGraph --separateProfileRun --useSpinWait --nvtxMode=verbose --loadEngine=dense/depthwise.trt --exportTimes=dense/times.json --exportProfile=dense/profile.json --exportLayerInfo=dense/layer_info.json --timingCacheFile=dense/timing_cache --plugins=${SO_PATH} | grep "GPU Compute Time:" > dense/gpu_compute_time.txt
fi
