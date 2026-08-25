#include <iostream>
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


        __shared__ half_t SmemA[32*16];
        __shared__ half_t SmemB[32*16];


        for(int i=threadIdx.x;i<32*16;i+=blockDim.x){
            SmemA[i] = A[i];
            SmemB[i] = B[i];
        }

        Tensor sA = make_tensor(make_smem_ptr(SmemA),Layout<Shape<_32,_16>,Stride<_16,_1>>{});
        Tensor sB = make_tensor(make_smem_ptr(SmemB),Layout<Shape<_32,_16>,Stride<_16,_1>>{});



        TiledMMA tiledmma = make_tiled_mma(SM80_16x8x16_F32F16F16F32_TN{},
            Layout<Shape<_2,_2>>{},
            Tile<_32,_32,_16>{});


            ThrMMA thr_mma = tiledmma.get_slice(threadIdx.x);


            Tensor tcrA = thr_mma.partition_fragment_A(sA);
            Tensor tcrB = thr_mma.partition_fragment_B(sB);

            Tensor tcgC = thr_mma.partition_C(gC);
            Tensor tcrC = thr_mma.partition_fragment_C(tcgC);
            clear(tcrC);




            Copy_Atom<SM75_U32x4_LDSM_N,half_t> copyAtom_A;
            Copy_Atom<SM75_U32x4_LDSM_N,half_t> copyAtom_B;



            TiledCopy tiledcopyA = make_tiled_copy_A(copyAtom_A,tiledmma);
            TiledCopy tiledcopyB = make_tiled_copy_B(coptAtom_B,tiledmma);


            ThrCopy thrcopyA = tiledcopyA.get_slice(threadIdx.x);
            ThrCopy thrcopyB = tiledcopyB.get_slice(threadIdx.x);


            Tensor txsA = thrcopyA.partition_S(sA);
            Tensor txsB = thrcopyB.partition_S(sB);


            Tensor txrA = thrcopyA.retile_D(tcrA);
            Tensor txrB = thrcopyB.retile_D(tcrB);



            copy(copyAtom_A,txsA,txrA);
            copy(copyAtom_B,txsB,txrB);


            gemm(tiledmma,tcrA,tcrB,tcrC);


            copy(tcrC,tcgC);






}



int main(){




    return 0;
}