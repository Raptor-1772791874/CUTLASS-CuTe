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
        constexpr int Stages = 2;


        __shared__ half_t SmemA[Stages][32*16];
        __shared__ half_t SmemB[Stages][32*16];


        auto Smem_layout = Layout<Shape<_32,_16>,Stride<_16,_1>>{};


        auto Smem_swizzle = composition(Swizzle<1,3,3>{},Smem_layout); 



        Tensor sA_0 = make_tensor(make_smem_ptr(SmemA[0]),Smem_swizzle);
        Tensor sA_1 = make_tensor(make_smem_ptr(SmemA[1]),Smem_swizzle);
        Tensor sB_0 = make_tensor(make_smem_ptr(SmemB[0]),Smem_swizzle);
        Tensor sB_1 = make_tensor(make_smem_ptr(SmemB[1]),Smem_swizzle);



        using G2SAtom = Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>,half_t>;

        TiledCopy g2s_copy = make_tiled_copy(G2SAtom{},Layout<Shape<_32,_2>,Stride<_2,_1>>{},Layout<Shape<_1,_8>>{});



         TiledMMA tiledmma = make_tiled_mma(SM80_16x8x16_F32F16F16F32_TN{},
            Layout<Shape<_2,_2>>{},
            Tile<_32,_32,_16>{});


            ThrMMA thr_mma = tiledmma.get_slice(threadIdx.x);


            Tensor tcrA = thr_mma.partition_fragment_A(sA_0);
            Tensor tcrB = thr_mma.partition_fragment_B(sB_0);

            Tensor tcgC = thr_mma.partition_C(gC);
            Tensor tcrC = thr_mma.make_fragment_C(tcgC);
            clear(tcrC);
           


            Copy_Atom<SM75_U32x4_LDSM_N,half_t> copyAtom_A;
            Copy_Atom<SM75_U32x4_LDSM_N,half_t> copyAtom_B;



            TiledCopy tiledcopyA = make_tiled_copy_A(copyAtom_A,tiledmma);
            TiledCopy tiledcopyB = make_tiled_copy_B(copyAtom_B,tiledmma);


            ThrCopy thrcopyA = tiledcopyA.get_slice(threadIdx.x);
            ThrCopy thrcopyB = tiledcopyB.get_slice(threadIdx.x);


            Tensor txsA_0 = thrcopyA.partition_S(sA_0);
            Tnesor txsA_1 = thrCopyA.partition_S(sA_1);
            Tensor txsB_0 = thrCopyA.partition_S(sB_0);
            Tensor txsB_1 = thrcopyB.partition_S(sB_1);

  

            Tensor txrA = thrcopyA.retile_D(tcrA);
            Tensor txrB = thrcopyB.retile_D(tcrB);


            //预取tile0
            Tensor gAtile0 = make_tensor(make_gmem_ptr(A),Layout<Shape<_32,_16>,Stride<_128,_1>>{});
            Tensor gBtile0 = make_tensor(make_gmem_ptr(B),Layout<Shape<_32,_16>,Stride<_128,_1>>{});


            if(threadIdx.x<64){


                ThrCopy g2s_tile0 = g2s_copy.getslice(threadIdx.x);
                Tensor tAgA_0 = g2s_tile0.partition_S(gAtile0)
                Tensor tAsA_0 = g2s_tile0.partition_D(sA_0);
 

                copy(g2scopy,tAgA_O,tAsA_0);

            }
            else{


                int lane = threadIdx.x-64;
                ThrCopy g2s_tile0 = g2s_copy.get_slice(lane);
                Tensor tBgB = g2s_tile0.partition_S(gBtile0);
                Tensor tBsB = g2s_tile0.partition_D(sB_0);
                
                
                copy(g2s_tile0,tBgB,tBsB);

            }


            cp_async_fence();
            cp_async_wait<0>();

            __syncthreads();



        for(int kt=0;kt<K/BK;++kt){


            int Compute_stage = kt & 1;
            int Next_stage = Compute_stage ^ 1;

            //它来回答是否有下一个块需要prefetch
            bool has_next = (kt+1 < K/BK);
 
            int next_k0 = (kt+1)*BK;


            if(has_next){

        Tensor gA_Next_tile = make_tensor(make_gmem_ptr(A+next_k0),Layout<Shape<_32,_16>,Stride<_128,_1>>{});
        Tensor gB_Next_tile = make_tensor(make_gmem_ptr(B+next_k0),Layout<Shape<_32,_16>,Stride<_128,_1>>{});



        if(threadIdx.x < 64){
            
            ThrCopy thr_g2s = g2s_copy.get_slice(threadIdx.x);
            Tensor tAgA = thr_g2s.partition_S(gA_Next_tile);


            if(Next_stage==0){
            Tensor tAsA = thr_g2s.partition_D(sA_0);
            copy(g2s_copy,tAgA,tAsA);
            }

            else{
            Tensor tAsA = thr_g2s.partition_D(sA_1);
            copy(g2s_copy,tAgA,tAsA);
            }
        }
        else{

            int copy_id = threadIdx.x - 64;

            ThrCopy thr_g2s = g2s_copy.get_slice(copy_id);
            Tensor tBgB = thr_g2s.partition_S(gB_Next_tile);

            
            if(Next_stage==0){
            Tensor tBsB = thr_g2s.partition_D(sB_0);
            copy(g2s_copy,tBgB,tBsB);
            }

            else{
            Tensor tBsB = thr_g2s.partition_D(sB_1);
            copy(g2s_copy,tBgB,tBsB);
            }

        }


            cp_async_fence();
        
        }

            

          if(Compute_stage==0){
            copy(copyAtom_A,txsA_0,txrA);
            copy(copyAtom_B,txsB_0,txrB);
          }
          else{
            copy(copyAtom_A,txsA_1,txrA);
            copy(copyAtom_B,txsB_1,txrB);
          }


            gemm(tiledmma,tcrA(_,_,Int<0>{}),tcrB(_,_,Int<0>{}),tcrC);



            //保证Barrier正确，如果是最后一次循环则不会进入触发sync
            if(has_next){

                cp_async_wait<0>();
                __syncthreads();

            }
        
        
        
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


    cout<<C[1]<<" "<<C[256]<<" "<<C[600]<<endl;


    cudaFree(device_A);
    cudaFree(device_B);
    cudaFree(device_C);
    free(A);
    free(B);
    free(C);













    return 0;
}
