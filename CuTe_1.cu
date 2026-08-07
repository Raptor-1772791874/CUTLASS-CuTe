#include <cuda_runtime.h>
#include <iostream>
#include <cute/tensor.hpp>

using namespace std;




int main(){
    //auto A = cute::Int<4>{};
    //auto B = cute::Int<8>{};
    //auto C = A*B;
    //cout<<int(C)<<endl;  //int()包裹住可以将其类型转回普通类型int再打印出值

    //auto tensor_A = cute::make_tensor(cute::make_gmem_ptr(A),cute::make_layout(cute::make_shape(M,K),cute::make_stride(K,1)));
    //auto tensor_B = cute::make_tensor(cute::make_gmem_ptr(B),cute::make_layout(cute::make_shape(K,N),cute::make_stride(N,1)));

    float data[8]={0,1,2,3,4,5,6,7};

    auto layout = make_layout(cute::make_shape(cute::Int<4>{},cute::Int<2>{}),cute::make_stride(cute::Int<2>{},cute::Int<1>{}));

    auto Tensor_A = make_tensor(data,layout);

    cou<<Tensor(1,2)<<endl;  //打印第1行第2列的值，即3

    auto Row_1 = Tensor_A(1,_);  //切到第一行来 行里的步长仍为1

    auto tile_A = cute::make_layout(cute::make_shape(cute::Int<8>{},cute::Int<8>{}),cute::make_stride(cute::Int<8>{},cute::Int<1>{}));

    auto tiler = cute::make_layout(cute::make_shape(cute::Int<4>{},cute::Int<4>{}),layoutRight{});  //自动以行主序填补Stride部分

    auto tile_11 = cute::zipped_divide(tile_A,tiler);

    auto new_tile_11 = tile_11(cute::make_coord(_,_),cute::make_coord(1,1));

    auto tiler_11 = cute::local_tile(tile_A,tiler,cute::make_coord(1,1));
    
}