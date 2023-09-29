#define WSIZE (98)
#define HSIZE (146)
#define CH (16)
#define SIZE (WSIZE*HSIZE*CH)
#define SM (48)
#define Tensorcore (8)

#include <stdio.h>
#include <cuda.h>
#include <mma.h>
#include <assert.h>
#include <cuda_fp16.h>
#include <random>
#include <iostream>
#include <cublas_v2.h>

// const half B[] = {
//         1,       0,       0,      0,      0,      0,      0,       0,
//         0,       1,      -1,   1./2,  -1./2,      2,     -2,      -1,
//   -21./4.,       1,       1,  1./4.,  1./4.,      4,      4,       0,
//         0, -17./4.,  17./4., -5./2.,  5./2., -5./2.,  5./2.,  21./4.,
//    21./4., -17./4., -17./4., -5./4., -5./4.,     -5,     -5,       0,
//         0,       1,      -1,      2,     -2,  1./2., -1./2., -21./4.,
//        -1,       1,       1,      1,      1,      1,      1,       0,
//         0,       0,       0,      0,      0,      0,      0,       1
// };

const unsigned B[] = {
    1006632960, 15360, 48128, 14336, 47104, 16384, 49152, 48128, 
    3309305856, 1006683200, 1006650432, 872464640, 872431872, 1140900096, 1140867328, 17728, 
    1161822208, 3292544000, 3292576768, 3170910208, 3170942976, 3305125888, 3305158656, 50496, 
    3154116608, 1006632960, 1006632960, 1006632960, 1006632960, 1006632960, 1006632960, 15360
};

const float G[] = {
       1,       0,      0,
  -2./9.,  -2./9., -2./9.,
  -2./9.,   2./9., -2./9.,
   1./90,   1./45,  2./45,
   1./90,  -1./45,  2./45,
  32./45,  16./45,  8./45,
  32./45, -16./45,  8./45,
       0,       0,      1
};

const float GT[] = {
    1,  -2./9.,  -2./9.,  1./90,  1./90,  32./45,  32./45,  0,
    0,  -2./9.,   2./9.,  1./45, -1./45,  16./45, -16./45,  0,
    0,  -2./9.,  -2./9.,  2./45,  2./45,   8./45,   8./45,  1
};


// const half A[] = {
//   1,     0,    0,     0,     0,      0, 0, 0,
//   1,     1,    1,     1,     1,      1, 0, 0,
//   1,    -1,    1,    -1,     1,     -1, 0, 0,
//   1,     2,    4,     8,    16,     32, 0, 0,
//   1,    -2,    4,    -8,    16,    -32, 0, 0,
//   1,  1./2, 1./4,  1./8, 1./16,  1./32, 0, 0,
//   1, -1./2, 1./4, -1./8, 1./16, -1./32, 0, 0,
//   0,     0,    0,     0,     0,      1, 0, 0
// };

const unsigned A[] = {
    1006648320, 15360, 15360, 15360, 15360, 15360, 0, 0, 
    1006648320, 3154132992, 1006650368, 3154135040, 1006652416, 3154137088, 0, 0, 
    1006648320, 3221239808, 1140864000, 3355455488, 1275079680, 3489671168, 0, 0, 
    1006632960, 3087007744, 872415232, 2952790016, 738197504, 2818587648, 0, 0
};



__device__ __forceinline__ unsigned int merge_half2_to_b32(half2 values)
{
  //===========merge two half into one .b32 register====================

  unsigned int merged_value;
  unsigned short *value_ptr = reinterpret_cast<unsigned short *>(&values);
  unsigned int upper_half = static_cast<unsigned int>(value_ptr[0]);
  unsigned int lower_half = static_cast<unsigned int>(value_ptr[1]);

  merged_value = (upper_half << 16) | lower_half;
  return merged_value;
}

//col compress
// __global__ void compress_B(half *ptr, int K, int N, unsigned *compressed_B)
// {
//     int x = threadIdx.x%8;
//     int y = threadIdx.x/8;
//     half2 values;
//     values.x = ptr[2 * y * N + x];
//     values.y = ptr[(2 * y + 1) * N + x];
//     compressed_B[y * N + x] = merge_half2_to_b32(values);
//     printf("%u, ", merge_half2_to_b32(values));
// }

void generate_random(half* data, int size, uint64_t seed) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(0, 1);
    for (int i = 0; i < size; i++) {
        data[i] = __float2half(dist(gen));
        // data[i] = i;
        // data[i] = i%16;
        // data[i] = 1;
    }
    return;
}

void calculate_GfGT(half* kernel, half *GfGT){
    float Gf[8*3*CH];
    for(int ch=0; ch<CH; ch++){
        for (int i = 0; i < 8; i++){
            for (int j = 0; j < 3; j++){
                float sum = 0.0f;
                for (int k = 0; k < 3; k++){
                    sum += G[i * 3 + k] * __half2float(kernel[k*3*CH + j*CH + ch]);
                }
                Gf[i*3*CH + j*CH + ch] = sum;
            }
        }
    }
    for(int ch=0; ch<CH; ch++){
        for (int i = 0; i < 8; i++){
            for (int j = 0; j < 8; j++){
                float sum = 0.0f;
                for (int k = 0; k < 3; k++){
                    sum += Gf[i*3*CH + k*CH + ch] * GT[k * 8 + j];
                }
                GfGT[i*8*CH + j*CH + ch] = __float2half(sum);
            }
        }
    }
}

__global__ void winograd(ushort4 *input, half *output, unsigned *B, half *GfGT, unsigned *A){
    using namespace nvcuda;
    
    __shared__ half s_output[6*6*8*CH];
    __shared__ half s_GfGT[8*8*CH];

    for(int i=threadIdx.x; i<8*8*CH; i+=blockDim.x*blockDim.y){
        s_GfGT[i] = GfGT[i];
    }
    //===================thread information================
    int laneid = threadIdx.x%32;
    int groupID = laneid >> 2;
    int threadID_in_group = laneid % 4;
    const int warpid = threadIdx.x >> 5;

    //coordinate information //1blockで(6×8)×6×chを計算(出力)している
    const int input_base= ((6*CH/4)*warpid+(48*CH/4)*blockIdx.x) + (6*WSIZE*CH/4)*blockIdx.y;
    const int output_base = (48*CH*blockIdx.x) + (6*(WSIZE-2)*CH)*blockIdx.y;
    const int s_output_base = 6*CH*(warpid);

    //===================load input and pack input and intra-thread transepose================
    unsigned in_temp[16];
    for(int i=0; i<4; i++){
        in_temp[4*i  ] = (unsigned )(input[input_base + laneid + (CH*WSIZE/4)*((2*(i+laneid%4))%8)].x << 16) | (unsigned)input[input_base + laneid + (CH*WSIZE/4)*((2*(i+laneid%4)+1)%8)].x;
        in_temp[4*i+1] = (unsigned )(input[input_base + laneid + (CH*WSIZE/4)*((2*(i+laneid%4))%8)].y << 16) | (unsigned)input[input_base + laneid + (CH*WSIZE/4)*((2*(i+laneid%4)+1)%8)].y;
        in_temp[4*i+2] = (unsigned )(input[input_base + laneid + (CH*WSIZE/4)*((2*(i+laneid%4))%8)].z << 16) | (unsigned)input[input_base + laneid + (CH*WSIZE/4)*((2*(i+laneid%4)+1)%8)].z;
        in_temp[4*i+3] = (unsigned )(input[input_base + laneid + (CH*WSIZE/4)*((2*(i+laneid%4))%8)].w << 16) | (unsigned)input[input_base + laneid + (CH*WSIZE/4)*((2*(i+laneid%4)+1)%8)].w;
    }
    //===================inter-tread transepose ================
    unsigned in[16];
    in[(0 + (laneid%4)*4)%16] = __shfl_sync(0xffffffff, in_temp[0 ],  laneid, 4);
    in[(1 + (laneid%4)*4)%16] = __shfl_sync(0xffffffff, in_temp[1 ],  laneid, 4);
    in[(2 + (laneid%4)*4)%16] = __shfl_sync(0xffffffff, in_temp[2 ],  laneid, 4);
    in[(3 + (laneid%4)*4)%16] = __shfl_sync(0xffffffff, in_temp[3 ],  laneid, 4);
    in[(12+ (laneid%4)*4)%16] = __shfl_sync(0xffffffff, in_temp[4 ],  (laneid+3), 4);
    in[(13+ (laneid%4)*4)%16] = __shfl_sync(0xffffffff, in_temp[5 ],  (laneid+3), 4);
    in[(14+ (laneid%4)*4)%16] = __shfl_sync(0xffffffff, in_temp[6 ],  (laneid+3), 4);
    in[(15+ (laneid%4)*4)%16] = __shfl_sync(0xffffffff, in_temp[7 ],  (laneid+3), 4);
    in[(8 + (laneid%4)*4)%16] = __shfl_sync(0xffffffff, in_temp[8 ],  (laneid+2), 4);
    in[(9 + (laneid%4)*4)%16] = __shfl_sync(0xffffffff, in_temp[9 ],  (laneid+2), 4);
    in[(10+ (laneid%4)*4)%16] = __shfl_sync(0xffffffff, in_temp[10],  (laneid+2), 4);
    in[(11+ (laneid%4)*4)%16] = __shfl_sync(0xffffffff, in_temp[11],  (laneid+2), 4);
    in[(4+ (laneid%4)*4)%16] = __shfl_sync(0xffffffff, in_temp[12],  (laneid+1), 4);
    in[(5+ (laneid%4)*4)%16] = __shfl_sync(0xffffffff, in_temp[13],  (laneid+1), 4);
    in[(6+ (laneid%4)*4)%16] = __shfl_sync(0xffffffff, in_temp[14],  (laneid+1), 4);
    in[(7+ (laneid%4)*4)%16] = __shfl_sync(0xffffffff, in_temp[15],  (laneid+1), 4);
    //===================winograd================
    unsigned reg[2];

    reg[0] = 0;
    reg[1] = 0;
    unsigned A1, A2, B1;
    for(int i=0; i<16/2; i++){
        reg[0] = 0;
        reg[1] = 0;
        //load 2*2 values of I
        A1 = in[2*i];
        A2 = in[2*i+1];
        //load 2*1 values of B
        B1 = B[threadID_in_group *8 + groupID];
        //=======================calc I' (IT * B)========================
        asm volatile(
            "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
            "{%0, %1},"
            "{%2, %3},"
            "{%4},"
            "{%5, %6};"
            : "=r"(reg[0]),
            "=r"(reg[1])
            : "r"(A1), "r"(A2),
            "r"(B1),
            "r"(reg[0]),
            "r"(reg[1]));
        //=======================transpose I' and packed into reg========================
        const int row = laneid>>2;
        const int col = laneid%4;
        unsigned temp0,temp1,temp2,temp3;
        temp0 = __shfl_sync(0xffffffff, reg[0], (2*col)*4+(row/2));
        temp1 = __shfl_sync(0xffffffff, reg[0], (2*col+1)*4+(row/2));
        temp2 = __shfl_sync(0xffffffff, reg[1], (2*col)*4+(row/2));
        temp3 = __shfl_sync(0xffffffff, reg[1], (2*col+1)*4+(row/2));
        if(row % 2 == 0){
            A1 = ((temp0 & 0xFFFF )<< 16) |  (temp1 & 0xFFFF);
            A2 = ((temp2 & 0xFFFF )<< 16) |  (temp3 & 0xFFFF);
        }else{
            A1 = (((temp0 >> 16) & 0xFFFF)<< 16) |  ((temp1 >> 16) & 0xFFFF);
            A2 = (((temp2 >> 16) & 0xFFFF)<< 16) |  ((temp3 >> 16) & 0xFFFF);
        }
        //=======================calc I (I'T * B)========================
        reg[0] = 0;
        reg[1] = 0;
        asm volatile(
            "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
            "{%0, %1},"
            "{%2, %3},"
            "{%4},"
            "{%5, %6};"
            : "=r"(reg[0]),
            "=r"(reg[1])
            : "r"(A1), "r"(A2),
            "r"(B1),
            "r"(reg[0]),
            "r"(reg[1]));
        //=======================element wise(calc O)========================
        //make_half2
        half2 O0,O1;
        O0.x = __hmul(__ushort_as_half((reg[0] >> 16) & 0xFFFF), s_GfGT[groupID*8*CH + (threadID_in_group * 2 + 1)*CH + 2*i]);
        O0.y = __hmul(__ushort_as_half(reg[0]& 0xFFFF), s_GfGT[groupID*8*CH + (threadID_in_group * 2)*CH + 2*i]);
        O1.x = __hmul(__ushort_as_half((reg[1] >> 16) & 0xFFFF), s_GfGT[groupID*8*CH + (threadID_in_group * 2 + 1)*CH + 2*i+1]);
        O1.y = __hmul(__ushort_as_half(reg[1]& 0xFFFF), s_GfGT[groupID*8*CH + (threadID_in_group * 2)*CH + 2*i+1]);
        // if(i == 1)printf("laneid=%d, %.1f, %.1f, %.1f, %.1f\n", laneid, (float)GfGT[groupID*8*CH + (threadID_in_group * 2 + 1)*CH + 2*i], (float)GfGT[groupID*8*CH + (threadID_in_group * 2)*CH + 2*i], (float)GfGT[groupID*8*CH + (threadID_in_group * 2 + 1)*CH + 2*i+1], (float)GfGT[groupID*8*CH + (threadID_in_group * 2)*CH + 2*i+1]);

        reg[0] = merge_half2_to_b32(O0);
        reg[1] = merge_half2_to_b32(O1);
        //=======================transpose O and packed into reg========================
        temp0 = __shfl_sync(0xffffffff, reg[0], (2*col)*4+(row/2));
        temp1 = __shfl_sync(0xffffffff, reg[0], (2*col+1)*4+(row/2));
        temp2 = __shfl_sync(0xffffffff, reg[1], (2*col)*4+(row/2));
        temp3 = __shfl_sync(0xffffffff, reg[1], (2*col+1)*4+(row/2));
        if(row % 2 == 0){
            A1 = ((temp0 & 0xFFFF )<< 16) |  (temp1 & 0xFFFF);
            A2 = ((temp2 & 0xFFFF )<< 16) |  (temp3 & 0xFFFF);
        }else{
            A1 = (((temp0 >> 16) & 0xFFFF)<< 16) |  ((temp1 >> 16) & 0xFFFF);
            A2 = (((temp2 >> 16) & 0xFFFF)<< 16) |  ((temp3 >> 16) & 0xFFFF);
        }
        //=======================calc O' (OT * A)========================
        reg[0] = 0;
        reg[1] = 0;
        B1 = A[threadID_in_group *8 + groupID];
        asm volatile(
            "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
            "{%0, %1},"
            "{%2, %3},"
            "{%4},"
            "{%5, %6};"
            : "=r"(reg[0]),
            "=r"(reg[1])
            : "r"(A1), "r"(A2),
            "r"(B1),
            "r"(reg[0]),
            "r"(reg[1]));
        //=======================transpose O' and packed into reg========================
        temp0 = __shfl_sync(0xffffffff, reg[0], (2*col)*4+(row/2));
        temp1 = __shfl_sync(0xffffffff, reg[0], (2*col+1)*4+(row/2));
        temp2 = __shfl_sync(0xffffffff, reg[1], (2*col)*4+(row/2));
        temp3 = __shfl_sync(0xffffffff, reg[1], (2*col+1)*4+(row/2));
        if(row % 2 == 0){
            A1 = ((temp0 & 0xFFFF )<< 16) |  (temp1 & 0xFFFF);
            A2 = ((temp2 & 0xFFFF )<< 16) |  (temp3 & 0xFFFF);
        }else{
            A1 = (((temp0 >> 16) & 0xFFFF)<< 16) |  ((temp1 >> 16) & 0xFFFF);
            A2 = (((temp2 >> 16) & 0xFFFF)<< 16) |  ((temp3 >> 16) & 0xFFFF);
        }
        //=======================calc O (O'T * A)========================
        reg[0] = 0;
        reg[1] = 0;
        asm volatile(
            "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
            "{%0, %1},"
            "{%2, %3},"
            "{%4},"
            "{%5, %6};"
            : "=r"(reg[0]),
            "=r"(reg[1])
            : "r"(A1), "r"(A2),
            "r"(B1),
            "r"(reg[0]),
            "r"(reg[1]));
        //========================================================================
        unsigned short c0,c1,c2,c3;
        c0 = reg[0]& 0xFFFF;
        c1 = (reg[0] >> 16) & 0xFFFF;
        c2 = reg[1]& 0xFFFF;
        c3 = (reg[1] >> 16) & 0xFFFF;
        if(threadID_in_group < 3 && groupID < 6){
            s_output[s_output_base + groupID*(6*8)*CH + (threadID_in_group * 2)*CH + 2*i] = __ushort_as_half(c0);
            s_output[s_output_base + groupID*(6*8)*CH + (threadID_in_group * 2 + 1)*CH + 2*i] = __ushort_as_half(c1);
            s_output[s_output_base + groupID*(6*8)*CH + (threadID_in_group * 2)*CH + 2*i+1] = __ushort_as_half(c2);
            s_output[s_output_base + groupID*(6*8)*CH + (threadID_in_group * 2 + 1)*CH + 2*i+1] = __ushort_as_half(c3);
            // printf("%.0f, ", __half2float(s_output[s_output_base + groupID*(6*4)*CH + (threadID_in_group * 2 + 1)*CH + 2*i+1]));
        }
    }
    __syncthreads();

    for(int i=0; i<6; i++){
        for(int j=threadIdx.x; j<(6*8)*CH; j+=blockDim.x*blockDim.y){
            output[output_base + j + i*(WSIZE-2)*CH] = s_output[j + i*(6*8)*CH];
        }
    }
    return;
}

void convolution_cpu(half *input, half *kernel, float *output){
    for(int ch=0; ch<CH; ch++){
        for(int i=0; i<HSIZE-2; i++){
            for(int j=0; j<WSIZE-2; j++){
                float sum = 0.0f;
                for(int k=0; k<3; k++){
                    for(int l=0; l<3; l++){
                        sum += __half2float(input[ch + (i+k)*WSIZE*CH + (j+l)*CH]) * __half2float(kernel[ch + k*3*CH + l*CH]);
                    }
                }
                output[ch + i*(WSIZE-2)*CH + j*CH] = sum;
            }
        }
    }
}

int main() {
    half h_input[SIZE], h_kernel[3*3*CH], h_output[(WSIZE-2)*(HSIZE-2)*CH];
    float CPU_output[(WSIZE-2)*(HSIZE-2)*CH];
    half GfGT[8*8*CH];
    half *d_output, *d_input;
    half *d_GfGT;
    unsigned *compressed_B, *compressed_A;
    cudaMalloc((void**)&d_input, (SIZE) * sizeof(half));
    cudaMalloc((void**)&d_output, (WSIZE-2)*(HSIZE-2)*CH  * sizeof(half));
    cudaMalloc((void **)&d_GfGT, 8 * 8 * CH * sizeof(half));
    cudaMalloc((void **)&compressed_B, (8 / 2) * 8 * sizeof(unsigned));
    cudaMalloc((void **)&compressed_A, (8 / 2) * 8 * sizeof(unsigned));

    //初期化
    generate_random(h_input, SIZE, 1);
    generate_random(h_kernel, 3*3*CH, 1);

    //calculate GfGT
    calculate_GfGT(h_kernel, GfGT);

    //転送
    cudaMemcpy(d_input, h_input, (SIZE) * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(compressed_B, B, 8 * 4 * sizeof(unsigned), cudaMemcpyHostToDevice);
    cudaMemcpy(compressed_A, A, 8 * 4 * sizeof(unsigned), cudaMemcpyHostToDevice);
    cudaMemcpy(d_GfGT, GfGT, 8 * 8 * CH * sizeof(half), cudaMemcpyHostToDevice);

    //depthwise convolution
    auto input_ushort4 = reinterpret_cast<ushort4*>(d_input);
    winograd<<<dim3(2,24), 256>>>(input_ushort4, d_output, compressed_B, d_GfGT, compressed_A);

    //calculate cpu
    for(int i=0; i<(WSIZE-2)*(HSIZE-2)*CH ; i++) CPU_output[i] = 0;
    convolution_cpu(h_input, h_kernel, CPU_output);

    //test
    cudaMemcpy(h_output, d_output, (WSIZE-2)*(HSIZE-2)*CH * sizeof(half), cudaMemcpyDeviceToHost);
    for(int i = 0; i < (WSIZE-2)*(HSIZE-2)*CH; i++){
        if(CPU_output[i]-0.1 > __half2float(h_output[i]) || __half2float(h_output[i]) > CPU_output[i]+0.1){
            printf("error\n");
            printf("i = %d %f %f\n",i, CPU_output[i], __half2float(h_output[i]));
            return 0;
        }
    }
    printf("success\n");
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_GfGT);
    cudaFree(compressed_B);
    cudaFree(compressed_A);

    return 0;
}