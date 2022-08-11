# GPU Accelerated Machine Learning Framework 

## Description

This project takes advantage of GPU acceleration to perform neural network computations extremely efficiently. The project has been developed in Visual Studio 2019, and uses the cuBLAS and CUDA libraries to achieve parallelization. Although we got very good results, this project was not created to beat the existing GPU accelerated programs that already exist, and it instead was created to explore the CUDA libraries and test our understanding of neural networks. So far, we have trained the model on the MNIST dataset, started work on GANs, and trained networks with >100,000,000 nodes in under an hour.

![ghdemon](https://user-images.githubusercontent.com/36086269/182450190-c23f7489-d321-4dc5-af15-cb2a07918984.gif)

## Installation

- Although it may work on non-Windows systems, this program has been designed specifically for Windows
- For this project to compile, you must install the latest version of the CUDA Toolkit, which can be found [here](https://developer.nvidia.com/cuda-downloads) 
  - The CUDA toolkit includes the two main libraries used: the core CUDA library as well as cuBLAS
- To directly access the code, this project can be cloned and directly loaded into Visual Studio 2019 for greatest simplicity

## Defining Neural Network Models

Because this project was created with experimentation in mind, it was our goal to only partially hide the complexities of model creation in a black box. While it is quite simple to instantiate a model with custom specs, it is also possible to add new functionality to the exposed backend code.

### Model creation

For now, model creation and functionality is contained within `layer.cpp` and the *FullyConnected* class. Although the file is named layer, the classes contained within it should be thought of as entire networks. Currently, the only class in layer is the *FullyConnected* class, which can be used to implement feedforward networks. The process of model creation begins with instantiating a FullyConnected layer. The following example shows the creation of a model using the ReLu activation function, a 0.015 learning rate, and 4 layers, which consist of 784, 10000, 10000, and 10 nodes respectively.

![image](https://user-images.githubusercontent.com/36086269/182448193-bbabcf4f-54cd-4f63-8ab7-d271915dd01f.png)

The creation of a model requires only a few characteristics of what the network should look like. The first argument requires a pointer based int array consisting of *layerNum* values, where `layers[i]` means that there will be *i* nodes/neurons in the i-th "layer" of the network once it is fully connected. While most of these values are up to user choice, `layers[0]` must also equal the size of the input data, and `layers[layerNum-1]` must equal the size of the expected output. The second argument is the integer *layerNum*, which stores the number of layers in the network(including input and output layers). The third argument is a floating point value that stores the learning rate of the network. The final argument is of the abstract type ActivationFunction, which is extended by the classes *Sigmoid*, *ReLu*, and *HypTan*.

### Training a model

One of the aspects of machine learning that we minimally incorporated into this project is the wide variety of data formats that one may want to process with a neural network. This project has no built in file format parsing, it is solely dedicated to the training of models once data has been read and pre-processed. The two functions useful for providing data to a model are:
```C++ 
float* feedForward(float* x);
void backProp(float* x, float* y);
```
In these two functions, *x* is taken to represent input, and *y* is taken to represent output. Running `feedForward(x)` will return the vector of the network's output layer after providing *x* to the input layer. Running `backProp(x,y)` will "train" or backpropagate on the model providing *x* to the input layer and comparing *y* to the actual network output. In other words, *y* is the expected output that the network should be outputting for the input *x*. It is important to note that the pointers x and y should be pointers to data stored in main memory(allocated by malloc), not to data stored in GPU memory(allocated by cudaMalloc). The functions will copy data to GPU memory as needed.

An example of providing data to a model:

![image](https://user-images.githubusercontent.com/36086269/182457979-029c9b25-2ea1-4295-a685-7c5ef37325e2.png)

### Custom activation functions:

Because of the use of object oriented design, it is fairly straightforward to add a custom activation function. While Sigmoid, ReLu, and HypTan cover many applications, there are certainly many others that could be necessary. All code relating to activation functions is stored in `ActivationFunctions.cu` and `ActivationFunctions.cuh`. While it may appear complex, integrating new activation functions can be done by copying and pasting most of the existing code for other activation functions. Firstly, the GPU must be informed how to parallelize the activation function through a kernel. In this context, a kernel is code that is compiled to run on GPU cores instead of traditional CPU cores(comparable to a shader used in processing graphics). The kernel implementation for the sigmoid kernel(and its derivative, sigmoid prime) is:

```CUDA
__device__ __forceinline__ float sigmoid(float x) {
	return 1.0 / (1.0 + exp(-x));
}
__device__ __forceinline__ float sigmoidPrime(float x) {
	float temp = sigmoid(x);
	return temp * (1 - temp);
}
__global__ void sigmoid_kernel(const float* __restrict src, float* __restrict__ dst, int len) {
	int stride = gridDim.x * blockDim.x;
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	for (int i = tid; i < len; i += stride) {
		dst[i] = sigmoid(src[i]);
	}
}
__global__ void sigmoidPrime_kernel(const float* __restrict src, float* __restrict__ dst, int len) {
	int stride = gridDim.x * blockDim.x;
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	for (int i = tid; i < len; i += stride) {
		dst[i] = sigmoidPrime(src[i]);
	}
}
```

When creating new activation functions, follow this template almost exactly. For a typical activation function, the only code you should need to change is the actual math operations that your activation function consists of. Now that the GPU has been instructed on how to handle the activation function, we must provide an interface for C++. In this project, this is done with a class that extends ActivationFunction, for example:

```C++
Sigmoid::Sigmoid() {
	name = "Sigmoid";
}
void Sigmoid::eval(float* x, float* y, int len) {
	dim3 dimBlock(256);
	int threadBlocks = (len + (dimBlock.x - 1)) / dimBlock.x;
	if (threadBlocks > 65520) threadBlocks = 65520;
	dim3 dimGrid(threadBlocks);
	sigmoid_kernel<<<dimGrid, dimBlock>>>(x, y, len);
}
void Sigmoid::evalPrime(float* x, float* y, int len) {
	dim3 dimBlock(256);
	int threadBlocks = (len + (dimBlock.x - 1)) / dimBlock.x;
	if (threadBlocks > 65520) threadBlocks = 65520;
	dim3 dimGrid(threadBlocks);
	sigmoidPrime_kernel<<<dimGrid, dimBlock>>>(x, y, len);
}
```

Once again, following this template will allow for a simplified experience adding your own activation functions.

