# Depthwise Convolution with Tensor Cores
以下の論文をDepthwise Convolutionに応用してみました。  
Liu, Junhong, Dongxu Yang, and Junjie Lai. "Optimizing winograd-based convolution with tensor cores." Proceedings of the 50th International Conference on Parallel Processing. 2021.  
https://dl.acm.org/doi/abs/10.1145/3472456.3472473  

## configure
- input: 98×146×16
- padding: 0
実装しやすいサイズで一旦試してみました。

## result
TensorRTので使用されるカーネルと比較しました  
|  |time[us]
| --- | --- |
| **my kernel(with tensor core)**  | 12.86  |
| **TensorRT** | 4.67  |  

あまりは速くならなかった..実装が良くない??  
プロファイラを見た感じあまり悪いところはなさそう  
tensor coreで使える型に変換したり、入れ替えたりすることのオーバーヘッドがdepthwiseでは大きく影響してしまうのかな

## 実行環境
- **GPU**: NVIDIA GeForce RTX 2080 SUPER
    - **SM数** 48
    - **Tensorコア数** 384 (8 per SM)
    - **CUDAコア数** 3072
- **GPUドライバ**: NVIDIA 535.104.05
- **CUDA**: 11.8
- **TensorRT**: 8.6.1
- **cuDNN**: 8.9.4
- **Python**: 3.10.12

## 参考リンク

- https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-1688
- https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-shuffle-functions
