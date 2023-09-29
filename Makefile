CP = 75

depthwise: depthwiseConvolution.cu
	nvcc -gencode=arch=compute_${CP},code=sm_${CP} --generate-line-info depthwiseConvolution.cu -o depthwiseConvolution

run:
	./depthwiseConvolution