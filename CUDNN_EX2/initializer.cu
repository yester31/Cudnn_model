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


using namespace std;
using namespace cv;


void InitWeightsXavier(
	const size_t NextLayerNodeNumber/*Number of output  feature maps*/, 
	const size_t PrevLayerNodeNumber/*Number of input feature maps*/,
	const size_t Height/*Height of each filter.*/, 
	const size_t Width/*Width of each filter.*/,
	float**** Weights)
{


	float sigma = sqrt(2 / (float)(PrevLayerNodeNumber + NextLayerNodeNumber));

	random_device rd;
	mt19937 gen(rd());
	normal_distribution<float> d(0, sigma);


	//Weights 정의
	for (int och = 0; och < NextLayerNodeNumber; och++)
	{
		for (int ch = 0; ch < PrevLayerNodeNumber; ch++)
		{
			for (int row = 0; row < Height; row++)
			{
				for (int col = 0; col < Width; col++)
				{
					Weights[och][ch][row][col] = d(gen);
				}
			}
		}
	}


}

void InitWeightsbias(const size_t & numOutSize, float* Weightsbias)
{
	for (int i = 0; i < numOutSize; i++)
	{
		Weightsbias[i] = 1.0f;
	}
}

int main() {

	

	const int filt_n = 2; /*Number of output feature maps*/ 
	const int filt_c = 2; /*Number of input feature maps*/
	const int filt_h = 4;
	const int filt_w = 4;

	// 4차 행렬 동적 할당 선언.
	float**** Weights = new float***[filt_n];
	for (int i = 0; i < filt_n; i++)
	{
		Weights[i] = new float**[filt_c];

		for (int j = 0; j < filt_c; j++)
		{
			Weights[i][j] = new float*[filt_h];

			for (int k = 0; k < filt_h; k++)
			{
				Weights[i][j][k] = new float[filt_w];
			}
		}
	}

	InitWeightsXavier(filt_n, filt_c, filt_h, filt_w, Weights);


	//Filter 정의
	for (int och = 0; och < filt_n; och++)
	{
		for (int ch = 0; ch < filt_c; ch++)
		{
			for (int row = 0; row < filt_h; row++)
			{
				for (int col = 0; col < filt_w; col++)
				{

					cout << setw(10) << Weights[och][ch][row][col] << " " ;
				}cout << endl;
			}cout << endl; cout << endl;
		}cout << endl; cout << endl;
	}



	// bias
	float* Bias = new float[filt_n];

	InitWeightsbias(filt_n, Bias);

	for (int col = 0; col < filt_n; col++)
	{
		cout << setw(10) << Bias[col] << " ";
	}cout << endl;

}