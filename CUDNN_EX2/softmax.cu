

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

	const int batch_size = 20;// 이미지수
	const int num_labels = 10; // 라벨 수 

	float target[batch_size]; // 정답
	float yhat[batch_size][num_labels] ; // soft max 결과 값 

	float *dev_target, *dev_yhat;


	// GPU 메모리를 할당한다 .
	cudaMalloc((void**)&dev_target, batch_size*num_labels * sizeof(float));
	cudaMalloc((void**)&dev_yhat, batch_size * num_labels * sizeof(float));

	
	// 임시 값 지정 (soft max 결과 값 )
	for (int j = 0 ; j < batch_size; j++)
	{
	    for (int i = 0; i < num_labels; i++)
	    {
	        yhat[j][i] = 1.0f;
	    }
	}

	// 임시 값 지정 (정답)
	for (int j = 0; j < batch_size; j++)
	{
	    target[j] = j % 10;
	}

	// 배열 GPU로 복사한다.
	cudaMemcpy(dev_target, target, batch_size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_yhat, yhat, batch_size * num_labels * sizeof(float), cudaMemcpyHostToDevice);

	// 커널 함수 ( 오차(=dloss=dy)값 계산) 
	SoftmaxLossBackprop <<<(batch_size +511)/512, 512 >> >(dev_target, num_labels, batch_size, dev_yhat );

	// dloss 값 GPU - > CPU로 복사
	cudaMemcpy(yhat, dev_yhat, batch_size * num_labels * sizeof(float), cudaMemcpyDeviceToHost);

	//dloss 결과를 화면에 출력한다.
	for (int j = 0; j < batch_size; j++) {
		for (int i = 0; i < num_labels; i++) {
			cout << yhat[j][i] << "  ";
		}cout << endl;
	}

	// GPU에 할당된 메모리를 해제한다.
	cudaFree(dev_target);
	cudaFree(dev_yhat);

	return 0;
}