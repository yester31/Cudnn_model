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


int main()
{
    int batchSize = 1;
	Mat img = imread("D:\\DataSet\\cifar\\test\\0_cat.png");
	unsigned char* imgd = img.data;
	size_t imgEleSize = img.elemSize();		// 한픽셀의 실제사이즈, rgb = 3
	size_t imgWidth = img.cols;				// 열 수, 이미지 가로 크기, 입력 데이터의 가로 길이
	size_t imgHeight = img.rows;			// 행의 수, 이미지 세로 크기, 입력 데이터의 세로 길이
	size_t imgChannel = img.channels();		// 깊이, 이미지 채널 수, 입력 데이터의 채널 수

											// 4차 행렬 동적 할당 선언.
	float**** Input = new float** *[batchSize];

	for (int i = 0; i < batchSize; i++)
	{
		Input[i] = new float**[3];

		for (int j = 0; j < 3; j++)
		{
			Input[i][j] = new float*[32];

			for (int k = 0; k < 32; k++)
			{
				Input[i][j][k] = new float[32];
			}
		}
	}

	for (int i = 0; i < batchSize; i++)
	{
		for (int c = 0; c < 3; c++)
		{
			for (int y = 0; y < 32; y++)
			{
				for (int x = 0; x < 32; x++)
				{
					Input[i][c][y][x] = imgd[3 * 32 * x + 3 * y + c];
					//Input[i][c][y][x] = temp[3 * 32 * y + 3 * x + c] * (1.0 / 255);
				}
			}
		}
	}


	vector<float> Input2;
	Input2.resize(batchSize * 3 * 32 * 32);


	for (int i = 0; i < batchSize; i++)
	{

		for (int c = 0; c < 3; c++)
		{
			for (int y = 0; y < 32; y++)
			{
				for (int x = 0; x < 32; x++)
				{
					Input2.push_back(Input[i][c][y][x]);
					
				}
			}
		}
	}



	for (int y = 0; y < batchSize * 3 * 32 * 32; y++)
	{
		cout << setw(3) << Input2[y] << "::";

		if ((y + 1) % 32 == 0)
		{
			cout << endl;
		}
	} cout << endl; cout << endl;
	
	/*

    // 4차 행렬 동적 할당 선언.
    float**** InputArray = new float***[batchSize];

    for (int i = 0; i < batchSize; i++)
    {
        InputArray[i] = new float** [3];

        for (int j = 0; j < 3; j++)
        {
            InputArray[i][j] = new float*[32];

            for (int k = 0; k < 32; k++)
            {
                InputArray[i][j][k] = new float[32];
            }
        }
    }

    for (int i = 0; i < batchSize; i++)
    {
        for (int c = 0; c < 3; c++)
        {
            for (int y = 0; y < 32; y++)
            {
                for (int x = 0; x < 32; x++)
                {
                    InputArray[i][c][y][x] = imgd[3 * 32 * y + 3 * x + c];
                    //Input[i][c][y][x] = temp[3 * 32 * y + 3 * x + c] * (1.0 / 255);
                }
            }
        }
    }

	


    for (int y = 0; y < 32; y++)
    {
        for (int x = 0; x < 32; x++)
        {
            cout << setw(3) << InputArray[0][0][y][x] << "::";
        } cout << endl;
    } cout << endl; cout << endl;


	cout << "====================================================4차 배열로 기존 방식 index를 이와 같이하여  3 * 32 * y + 3 * x + c 가져오기" << endl;

	// 4차 행렬 동적 할당 선언.
	float**** InputArray2 = new float***[batchSize];

	for (int i = 0; i < batchSize; i++)
	{
		InputArray2[i] = new float**[3];

		for (int j = 0; j < 3; j++)
		{
			InputArray2[i][j] = new float*[32];

			for (int k = 0; k < 32; k++)
			{
				InputArray2[i][j][k] = new float[32];
			}
		}
	}

	for (int i = 0; i < batchSize; i++)
	{
		for (int c = 0; c < 3; c++)
		{
			for (int y = 0; y < 32; y++)
			{
				for (int x = 0; x < 32; x++)
				{
					InputArray2[i][c][y][x] = imgd[3 * 32 * x + 3 * y + c];
					//Input[i][c][y][x] = temp[3 * 32 * y + 3 * x + c] * (1.0 / 255);
				}
			}
		}
	}

	for (int y = 0; y < 32; y++)
	{
		for (int x = 0; x < 32; x++)
		{
			cout << setw(3) << InputArray2[0][0][y][x] << "::";
		} cout << endl;
	} cout << endl; cout << endl;

	cout << "====================================================4차 배열로 기존 방식 index를 이와 같이하여  3 * 32 * x + 3 * y + c 가져오기" << endl;


	
	*/




    cudnnHandle_t cudnn;
    cublasHandle_t cublasHandle;
    cublasCreate(&cublasHandle);
    float alpha = 1.0;
    float beta = 0.0;
}