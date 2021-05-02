

#include <iomanip>
#include <iostream>
#include <cstdlib>
#include <math.h>
#include <cuda.h>
#include <cudnn.h>
#include <stdio.h>
#include <string.h>
#include <opencv2/opencv.hpp>
#include <io.h>
#include <vector>
#include "cuda_runtime.h"
#include <device_launch_parameters.h>
#include <cublas_v2.h>
#include <random>


//***********************
//**SoftmaxLossBackprop**
//***********************

using namespace std;
using namespace cv;



__global__ void SoftmaxLossBackprop(const float *label, const int num_labels, const int batch_size, float *diff)
{

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	
	if (idx >= batch_size)
		return;
	
	const int label_value = static_cast<int>(label[idx]);

	// For each item in the batch, decrease the result of the label's value by 1
	diff[idx * num_labels + label_value] -= 1.0f;
}

int main() {

	const int batch_size = 20;// �̹�����
	const int num_labels = 10; // �� �� 

	float target[batch_size]; // ����
	float yhat[batch_size][num_labels] ; // soft max ��� �� 

	float *dev_target, *dev_yhat;


	// GPU �޸𸮸� �Ҵ��Ѵ� .
	cudaMalloc((void**)&dev_target, batch_size*num_labels * sizeof(float));
	cudaMalloc((void**)&dev_yhat, batch_size * num_labels * sizeof(float));

	
	// �ӽ� �� ���� (soft max ��� �� )
	for (int j = 0 ; j < batch_size; j++)
	{
	    for (int i = 0; i < num_labels; i++)
	    {
	        yhat[j][i] = 1.0f;
	    }
	}

	// �ӽ� �� ���� (����)
	for (int j = 0; j < batch_size; j++)
	{
	    target[j] = j % 10;
	}

	// �迭 GPU�� �����Ѵ�.
	cudaMemcpy(dev_target, target, batch_size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_yhat, yhat, batch_size * num_labels * sizeof(float), cudaMemcpyHostToDevice);

	// Ŀ�� �Լ� ( ����(=dloss=dy)�� ���) 
	SoftmaxLossBackprop <<<(batch_size +511)/512, 512 >> >(dev_target, num_labels, batch_size, dev_yhat );

	// dloss �� GPU - > CPU�� ����
	cudaMemcpy(yhat, dev_yhat, batch_size * num_labels * sizeof(float), cudaMemcpyDeviceToHost);

	//dloss ����� ȭ�鿡 ����Ѵ�.
	for (int j = 0; j < batch_size; j++) {
		for (int i = 0; i < num_labels; i++) {
			cout << yhat[j][i] << "  ";
		}cout << endl;
	}

	// GPU�� �Ҵ�� �޸𸮸� �����Ѵ�.
	cudaFree(dev_target);
	cudaFree(dev_yhat);

	return 0;
}