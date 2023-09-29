
#!/bin/bash

if [ -f "dense/depthwise.trt" ] && [ ! -f "dense/profile_depthwise.ncu-rep" ]; then
   /opt/nvidia/nsight-compute/2022.3.0/ncu -o dense/profile_depthwise --set full /usr/src/tensorrt/bin/trtexec  --loadEngine=dense/depthwise.trt --warmUp=0 --duration=0 --iterations=1 --useSpinWait --noDataTransfers --useCudaGraph
fi
