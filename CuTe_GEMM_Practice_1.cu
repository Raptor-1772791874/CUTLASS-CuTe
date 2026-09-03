//#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/algorithm/copy.hpp>

using namespace std;
using namespace cute;


 //  A/B [32][16] C[32][32]


__global__ void cute_gemm(half_t* const A,
    half_t* const B,
    float* C){

        Tensor gA = make_tensor(make_gmem_ptr(A),Layout<Shape<_32,_16>,Stride<_16,_1>>{});
        Tensor gB = make_tensor(make_gmem_ptr(B),Layout<Shape<_32,_16>,Stride<_16,_1>>{});
        Tensor gC = make_tensor(make_gmem_ptr(C),Layout<Shape<_32,_32>,Stride<_32,_1>>{});


        auto SmemA_layout = Layout<Shape<_32,_16>,Stride<_16,_1>>{};


        __shared__ half_t SmemA[32*16];
        __shared__ half_t SmemB[32*16];


        auto Smem_ptr = composition(Swizzle<1,3,3>{},SmemA_layout);


        Tensor sA = make_tensor(make_smem_ptr(SmemA),Smem_ptr);
        Tensor sB = make_tensor(make_smem_ptr(SmemB),Layout<Shape<_32,_16>,Stride<_16,_1>>{});



        for(int i=threadIdx.x;i<32*16;i+=blockDim.x){
            int r = i/16;
            int c = i%16;
            
            sA(r,c) = gA(r,c);
            SmemB[i] = B[i];
        }

        __syncthreads();

        



        TiledMMA tiledmma = make_tiled_mma(SM80_16x8x16_F32F16F16F32_TN{},
            Layout<Shape<_2,_2>>{},
            Tile<_32,_32,_16>{});


            ThrMMA thr_mma = tiledmma.get_slice(threadIdx.x);


            Tensor tcrA = thr_mma.partition_fragment_A(sA);
            Tensor tcrB = thr_mma.partition_fragment_B(sB);

            Tensor tcgC = thr_mma.partition_C(gC);
            Tensor tcrC = thr_mma.make_fragment_C(tcgC);
            clear(tcrC);
           



            Copy_Atom<SM75_U32x4_LDSM_N,half_t> copyAtom_A;
            Copy_Atom<SM75_U32x4_LDSM_N,half_t> copyAtom_B;



            TiledCopy tiledcopyA = make_tiled_copy_A(copyAtom_A,tiledmma);
            TiledCopy tiledcopyB = make_tiled_copy_B(copyAtom_B,tiledmma);


            ThrCopy thrcopyA = tiledcopyA.get_slice(threadIdx.x);
            ThrCopy thrcopyB = tiledcopyB.get_slice(threadIdx.x);


            Tensor txsA = thrcopyA.partition_S(sA);
            Tensor txsB = thrcopyB.partition_S(sB);

            if(threadIdx.x==0){
                printf("txsA:");
                print(txsA);
                printf("\n");
                printf("txsB");
                print(txsB);
            }

            int lane = threadIdx.x % 32;
            int warp = threadIdx.x / 32;

    if (warp == 0) {
    auto p = raw_pointer_cast(txsA.data());

    // 转成 shared memory address
    unsigned int addr =
        static_cast<unsigned int>(__cvta_generic_to_shared(p));

    unsigned int base =
        static_cast<unsigned int>(__cvta_generic_to_shared(SmemA));

    int byte_offset = int(addr - base);
    int elem_offset = byte_offset / sizeof(half_t);
    int bank        = (byte_offset / 4) % 32;

    printf(
        "lane:%2d  elem:%3d  byte:%3d  bank:%2d\n",
        lane,
        elem_offset,
        byte_offset,
        bank
    );
}
  


            Tensor txrA = thrcopyA.retile_D(tcrA);
            Tensor txrB = thrcopyB.retile_D(tcrB);



            copy(copyAtom_A,txsA,txrA);
            copy(copyAtom_B,txsB,txrB);


            gemm(tiledmma,tcrA(_,_,Int<0>{}),tcrB(_,_,Int<0>{}),tcrC);


            copy(tcrC,tcgC);






}



int main(){

    int N = 32*16;
    half_t* A;
    half_t* B;
    float* C;

    size_t range = sizeof(half_t) * N;
    size_t range_C = sizeof(float) * 32*32;


    A = (half_t*)malloc(range);
    B = (half_t*)malloc(range);
    C = (float*)malloc(range_C);


    for(int i=0;i<N;++i){
        A[i] = 1.0f;
        B[i] = 1.0f;

    }



    half_t *device_A , *device_B ; 
    float* device_C;
    cudaMalloc((void**)&device_A,range);
    cudaMalloc((void**)&device_B,range);
    cudaMalloc((void**)&device_C,range_C);


    cudaMemcpy(device_A,A,range,cudaMemcpyHostToDevice);
    cudaMemcpy(device_B,B,range,cudaMemcpyHostToDevice);




    cute_gemm<<<1,128>>>(device_A,device_B,device_C);




    cudaError_t err = cudaGetLastError();

    if(err!=cudaSuccess){
        cout<< "Kernel launch error: " << cudaGetErrorString(err) << endl;
     return EXIT_FAILURE;
    }

    cudaDeviceSynchronize();


    cudaMemcpy(C,device_C,range_C,cudaMemcpyDeviceToHost);


    //cout<<C[1]<<" "<<C[23]<<endl;


    cudaFree(device_A);
    cudaFree(device_B);
    cudaFree(device_C);
    free(A);
    free(B);
    free(C);













    return 0;
}