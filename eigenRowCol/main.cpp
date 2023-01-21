#include "iostream"
#include <vector>
#include <stdlib.h>
#include "/mnt/e/College/eigen/Eigen/Dense"
#include "/mnt/e/College/eigen/unsupported/Eigen/CXX11/Tensor"

typedef Eigen::Matrix<signed char, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixR8i;
typedef Eigen::Matrix<signed char, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> MatrixC8i;

typedef Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixR32i;
typedef Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> MatrixC32i;

// References
// https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D
// https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html

//batches:2, channels: 2, rows: 5, cols: 5
int input[100] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67,
68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84,
85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99};

// batches:3, channels: 2, rows: 3, cols: 3
int kern[54] = {1,1,1,1,1,1,1,1,1,1,
                1,1,1,1,1,1,1,1,1,1,
                1,1,1,1,1,1,1,1,1,1,
                1,1,1,1,1,1,1,1,1,1,
                1,1,1,1,1,1,1,1,1,1,
                1,1,1,1};

// SAME
// batches:2, channels: 3, rows: 5, cols: 5
// int output[150] = {0};

// VALID
// batches:2, channels: 3, rows: 3, cols: 3
int output[54] = {0};


void create_matrices(void)
{
    // 1 2
    // 3 4
    std::vector<signed char> vecA = {1,2,3,4}; 

    // 1 0
    // 2 4
    std::vector<signed char> vecB = {1,2,0,4}; // This is truly column major
    
    // Row
    Eigen::Map<MatrixR8i> map_row = Eigen::Map<MatrixR8i>(vecA.data(),2,2); 
    MatrixR32i mat_rowcast = map_row.template cast<int>();
    std::cout << "Row major Vector A" << std::endl;
    for(int i =0; i< mat_rowcast.size();i++)
    {
        std::cout << *(mat_rowcast.data()+i) << " ";
    }
    std::cout << std::endl << std::endl;

    // Column 
    Eigen::Map<MatrixR8i> map_col = Eigen::Map<MatrixR8i>(vecB.data(),2,2); 
    MatrixR32i mat_colcast = map_col.template cast<int>();
    std::cout << "Col major Vector B" << std::endl;
    for(int i =0; i<mat_colcast.size();i++)
    {
        std::cout << *(mat_colcast.data()+i) << " ";
    }
    std::cout << std::endl << std::endl;
    //std::cout << "Column matrix block form: \n";
    //std::cout << mat_colcast << std::end << std::endl;

    // Result as Row Major
    std::cout << "Row result: " << std::endl;
    MatrixR32i res_row = mat_rowcast*mat_colcast;
    for(int i =0; i<res_row.size();i++)
    {
        std::cout << *(res_row.data()+i) << " ";
    }
    std::cout << std::endl << std::endl;

    // Result as Col 
    std::cout << "Col result, row VecA * col VecB" << std::endl;
    MatrixC32i res_col = mat_rowcast*mat_colcast;
    std::cout << "Row Major? " << (MatrixC32i::IsRowMajor) << std::endl;
    for(int i =0; i<res_col.size();i++)
    {
        std::cout << *(res_col.data()+i) << " ";
    }
    std::cout << std::endl << std::endl;

    // Recast as Row Major
    std::cout << "Back to row" << std::endl;
    Eigen::Map<MatrixR32i> map_back_to_row = Eigen::Map<MatrixR32i>(res_col.data(),2,2); 
    std::cout << "Row Major? " << (MatrixR32i::IsRowMajor) << std::endl;
    for(int i =0; i< map_back_to_row.size();i++)
    {
        std::cout << *(map_back_to_row.data()+i) << " ";
    }
    std::cout << std::endl << std::endl;
    return;
}

void tensor_fun(void)
{
    int kern_w, kern_h, stride_w, stride_h, dilation_w, dilation_h;
    kern_w = 3;
    kern_h = 3;
    stride_w = 1;
    stride_h = 1;
    dilation_w = 1;
    dilation_h = 1;
    Eigen::PaddingType padding = Eigen::PADDING_SAME;

    int patch_count = 2*5*5;
    int kern_channels = 3; 
    int kern_count = 2;

    Eigen::Tensor<int, 4> inputTensor(2, 3, 5, 5);
    Eigen::Tensor<int, 4> kernelTensor(kern_count, kern_channels, kern_h, kern_w);

    kernelTensor = kernelTensor.reshape(Eigen::array<int, 2>({kern_channels*kern_w*kern_h, kern_count}));
    Eigen::array<Eigen::IndexPair<int>, 1> contract_dims = {Eigen::IndexPair<int>(1, 0)};

    auto outputTensor = inputTensor
            .extract_image_patches(kern_w, kern_h, stride_w, stride_h, dilation_w, dilation_h, padding)
            .reshape(Eigen::array<int, 2>({patch_count, kern_w*kern_h*kern_channels}));
//            .contract(kernelTensor, contract_dims);
//            .reshape(Eigen::array<int, 3>({ output_w, output_h, kern_count }));
    outputTensor.eval();
    const auto& d = outputTensor.dimensions();
    std::cout << std::endl;
    for(auto dim: d)
    {
        std::cout << "dim is: " << dim << ", ";
    }
    
}

void tensor_stable(void)
{

Eigen::array<Eigen::IndexPair<Eigen::Index>, 1> contract_dims;
contract_dims[0] = Eigen::IndexPair<Eigen::Index>(1, 0);

Eigen::array<Eigen::Index, 4> in_dims({2, 2, 5, 5});

// SAME
Eigen::array<Eigen::Index, 4> out_dims({2, 3, 5, 5});

// VALID
// Eigen::array<Eigen::Index, 4> out_dims({2, 3, 3, 3});


Eigen::array<Eigen::Index, 4> kernel_dims({3, 2, 3, 3});


Eigen::DSizes<Eigen::Index, 2> pre_contract_dims;
pre_contract_dims[1] = kernel_dims[2] * kernel_dims[1] * kernel_dims[0];
pre_contract_dims[0] = out_dims[2] * out_dims[3];
for (int i = 0; i < 1; ++i) {
  pre_contract_dims[0] *= in_dims[i];
}

Eigen::DSizes<Eigen::Index, 4> post_contract_dims;
post_contract_dims[3] = kernel_dims[0];
post_contract_dims[2] = out_dims[3];
post_contract_dims[1] = out_dims[2];
for (int i = 0; i < 1; ++i) {
  post_contract_dims[i] = in_dims[i];
}

Eigen::DSizes<Eigen::Index, 2> new_kernel_dims;
new_kernel_dims[0] = kernel_dims[2] * kernel_dims[1] * kernel_dims[0];
new_kernel_dims[1] = kernel_dims[3];

// float input0[50*15*200*3] = {0};
// float output0[50*11*3*200] = {0};
// float input1[5*200*3*200] = {0};

Eigen::TensorMap<Eigen::Tensor<int, 4, Eigen::RowMajor>>
    in(static_cast<int *>(input), in_dims),
    // out(static_cast<int *>(output), out_dims),
    kernel(static_cast<int *>(kern), kernel_dims);

auto out = in
    .extract_image_patches(kernel_dims[3], kernel_dims[2], 1,
                           1, 1, 1,
                           Eigen::PADDING_SAME)
    .reshape(pre_contract_dims)
    .contract(kernel.reshape(new_kernel_dims), contract_dims, Eigen::NoOpOutputKernel())
    .reshape(post_contract_dims);
    Eigen::Tensor<int, 4, Eigen::RowMajor> result = out;
    std::cout << "in:\n" << in.format(Eigen::TensorIOFormat::Plain()) << std::endl;
    std::cout << "kernel:\n" << kernel.format(Eigen::TensorIOFormat::Plain()) << std::endl;
    std::cout << "out:\n" << result.format(Eigen::TensorIOFormat::Plain()) << std::endl;
}


void tensor_small(void)
{

int input_small[36] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
34, 35};

int kern_small[24] = {1,1,1,1,1,1,1,1,1,1,
                      1,1,1,1,1,1,1,1,1,1,
                      1,1,1,1};

int output_small[24] = {0};

Eigen::array<Eigen::IndexPair<Eigen::Index>, 1> contract_dims;
contract_dims[0] = Eigen::IndexPair<Eigen::Index>(1, 0);

// orig
// Eigen::array<Eigen::Index, 4> in_dims({2, 2, 3, 3});

// * The input parameter is expected to be a tensor with a rank of 3 or more
// * (channels, height, width, and optionally others) 
Eigen::array<Eigen::Index, 4> in_dims({2, 2, 3, 3});

// SAME
// Eigen::array<Eigen::Index, 4> out_dims({2, 3, 2, 2});

// VALID
Eigen::array<Eigen::Index, 4> out_dims({3, 2, 2, 2});


// The kernel parameter is expected to be a 4D tensor (filters, channels,
// kernel_height, kernel_width)
Eigen::array<Eigen::Index, 4> kernel_dims({3, 2, 2, 2});


// tf ordering for patches to work
Eigen::array<Eigen::Index, 4> in_dims_perm({2, 3, 3, 2});
Eigen::array<Eigen::Index, 4> out_dims_perm({2, 2, 2, 3});


Eigen::DSizes<Eigen::Index, 2> pre_contract_dims;
pre_contract_dims[1] = kernel_dims[3] * kernel_dims[2] * kernel_dims[1];
pre_contract_dims[0] = out_dims_perm[1] * out_dims_perm[2];
for (int i = 0; i < 1; ++i) {
  pre_contract_dims[0] *= in_dims_perm[i];
}

Eigen::DSizes<Eigen::Index, 4> post_contract_dims;
post_contract_dims[3] = kernel_dims[0];
post_contract_dims[2] = out_dims_perm[2];
post_contract_dims[1] = out_dims_perm[1];
for (int i = 0; i < 1; ++i) {
  post_contract_dims[i] = in_dims_perm[i];
}

Eigen::DSizes<Eigen::Index, 2> new_kernel_dims;
new_kernel_dims[0] = kernel_dims[3] * kernel_dims[2] * kernel_dims[1];
new_kernel_dims[1] = kernel_dims[0];

Eigen::TensorMap<Eigen::Tensor<int, 4, Eigen::RowMajor>>
    in(static_cast<int *>(input_small), in_dims),
    // out(static_cast<int *>(output), out_dims),
    kernel(static_cast<int *>(kern_small), kernel_dims);

Eigen::array<Eigen::Index, 4> shuffles({0, 2, 3, 1});
Eigen::array<Eigen::Index, 4> shuffle_back({0, 3, 1, 2});
// Eigen::array<Eigen::Index, 4> shuffle_back({3, 0, 1, 2});
// Eigen::array<Eigen::Index, 4> shuffle_perm({1, 0, 2, 3});
Eigen::Tensor<int, 4, Eigen::RowMajor> in_perm = in.shuffle(shuffles);

auto out = in_perm
    .extract_image_patches(kernel_dims[2], kernel_dims[3], 1,
                           1, 1, 1,
                           Eigen::PADDING_VALID)
    .reshape(pre_contract_dims)
    .contract(kernel.reshape(new_kernel_dims), contract_dims, Eigen::NoOpOutputKernel())
    .reshape(post_contract_dims);
    Eigen::Tensor<int, 4, Eigen::RowMajor> result = out.shuffle(shuffle_back);// .shuffle(shuffle_perm);

    std::cout << "in:\n" << in.format(Eigen::TensorIOFormat::Plain()) << std::endl;
    std::cout << "in_perm:\n" << in_perm.format(Eigen::TensorIOFormat::Plain()) << std::endl;
    std::cout << "kernel:\n" << kernel.format(Eigen::TensorIOFormat::Plain()) << std::endl;
    std::cout << "out:\n" << out.eval() << std::endl;
    std::cout << "result:\n" << result.format(Eigen::TensorIOFormat::Plain()) << std::endl;
    //std::cout << "out dims:\n" << result.dimension(0) << "," << result.dimension(1) << "," << result.dimension(2) << "," << result.dimension(3) << "," << result.dimension(4) << std::endl;
//    std::cout << "out:\n" << result.format(Eigen::TensorIOFormat::Plain()) << std::endl;
}


int main()
{
    // create_matrices();
    // tensor_fun();
    // tensor_stable();
    tensor_small();
    return 0;
}