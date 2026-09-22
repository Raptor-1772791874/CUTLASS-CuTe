//#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/algorithm/copy.hpp>
#include <cute/arch/copy_sm80.hpp>

using namespace std;
using namespace cute;


 //  A/B [32][16] C[32][32]


__global__ void cute_gemm(half_t* const A,
    half_t* const B,
    float* C){

        Tensor gA = make_tensor(make_gmem_ptr(A),Layout<Shape<_32,_128>,Stride<_128,_1>>{});
        Tensor gB = make_tensor(make_gmem_ptr(B),Layout<Shape<_32,_128>,Stride<_128,_1>>{});
        Tensor gC = make_tensor(make_gmem_ptr(C),Layout<Shape<_32,_32>,Stride<_32,_1>>{});


        constexpr int K = 128;
        constexpr int BK = 16;

        clear(tcrC);



        for(int k0=0;k0<K;k0+=BK){


        Tensor gA_tile = make_tensor(make_geme_ptr(A+k0),Layout<Shape<_32,_16>,Stride<_128,_1>>{});
        Tensor gB_tile = make_tensor(make_gmem_ptr(B+k0),Layout<Shape<_32,_16>,Stride<_128,_1>>{});


        auto SmemA_layout = Layout<Shape<_32,_16>,Stride<_16,_1>>{};
        auto SmemB_layout = Layout<Shape<_32,_16>,Stride<_16,_1>>{};


        __shared__ half_t SmemA[32*16];
        __shared__ half_t SmemB[32*16];


        auto Smem_ptr = composition(Swizzle<1,3,3>{},SmemA_layout);
        auto Smem_ptr_1 = composition(Swizzle<1,3,3>{},SmemB_layout); 


        Tensor sA = make_tensor(make_smem_ptr(SmemA),Smem_ptr);
        Tensor sB = make_tensor(make_smem_ptr(SmemB),Smem_ptr_1);



        using G2SAtom = Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>,half_t>;


        TiledCopy g2s_copy = make_tiled_copy(G2SAtom{},Layout<Shape<_32,_2>,Stride<_2,_1>>{},Layout<Shape<_1,_8>>{});


        if(threadIdx.x < 64){
            
            ThrCopy thr_g2s = g2s_copy.get_slice(threadIdx.x);

            Tensor tAgA = thr_g2s.partition_S(gA_tile);
            Tensor tAsA = thr_g2s.partition_D(sA);

            copy(g2s_copy,tAgA,tAsA);

        }
        else{

            int copy_id = threadIdx.x - 64;

            ThrCopy thr_g2s = g2s_copy.get_slice(copy_id);

            Tensor tBgB = thr_g2s.partition_S(gB_tile);
            Tensor tBsB = thr_g2s.partition_D(sB);

            copy(g2s_copy,tBgB,tBsB);
        }



        cp_async_fence();
        cp_async_wait<0>();
      


        __syncthreads();

        



        TiledMMA tiledmma = make_tiled_mma(SM80_16x8x16_F32F16F16F32_TN{},
            Layout<Shape<_2,_2>>{},
            Tile<_32,_32,_16>{});


            ThrMMA thr_mma = tiledmma.get_slice(threadIdx.x);


            Tensor tcrA = thr_mma.partition_fragment_A(sA);
            Tensor tcrB = thr_mma.partition_fragment_B(sB);

            Tensor tcgC = thr_mma.partition_C(gC);
            Tensor tcrC = thr_mma.make_fragment_C(tcgC);
           



            Copy_Atom<SM75_U32x4_LDSM_N,half_t> copyAtom_A;
            Copy_Atom<SM75_U32x4_LDSM_N,half_t> copyAtom_B;



            TiledCopy tiledcopyA = make_tiled_copy_A(copyAtom_A,tiledmma);
            TiledCopy tiledcopyB = make_tiled_copy_B(copyAtom_B,tiledmma);


            ThrCopy thrcopyA = tiledcopyA.get_slice(threadIdx.x);
            ThrCopy thrcopyB = tiledcopyB.get_slice(threadIdx.x);


            Tensor txsA = thrcopyA.partition_S(sA);
            Tensor txsB = thrcopyB.partition_S(sB);


            //查找BankConflict，据结果设计Swizzle
            /*if(threadIdx.x==0){
                printf("txsA:");
                print(txsA);
                printf("\n");
                printf("txsB");
                print(txsB);
            }

            int lane = threadIdx.x % 32;
            int warp = threadIdx.x / 32;*/

    /*if (warp == 0) {
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
}  */
  


            Tensor txrA = thrcopyA.retile_D(tcrA);
            Tensor txrB = thrcopyB.retile_D(tcrB);



            copy(copyAtom_A,txsA,txrA);
            copy(copyAtom_B,txsB,txrB);


            gemm(tiledmma,tcrA(_,_,Int<0>{}),tcrB(_,_,Int<0>{}),tcrC);
        
        
            __syncthreads();
        }


            copy(tcrC,tcgC);






}



int main(){

    int N = 32*128;
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