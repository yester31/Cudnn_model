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

void checkCUDNN(cudnnStatus_t status)
{
	if (status != CUDNN_STATUS_SUCCESS)
		std::cout << "[ERROR] CUDNN " << status << std::endl;
}

int main()
{


	 //입력변수
	const int ImageNum = 1;
	const int FeatureNum = 3;
	const int FeatureHeight = 8;
	const int FeatureWidth = 8;

	// 4차 행렬 동적 할당 선언.
	float Input[ImageNum][FeatureNum][FeatureHeight][FeatureWidth];

	int count = 1;
	// mat 형식 - > 4차 행렬
	for (int i = 0; i < ImageNum; i++)
	{
		for (int c = 0; c < FeatureNum; c++)
		{
			for (int y = 0; y < FeatureHeight; y++)
			{
				for (int x = 0; x < FeatureWidth; x++)
				{
					
					//Input[i][c][y][x] = count++;
					Input[i][c][y][x] = 1;
				}
			}
		}
	}
	
	// input 4차 배열 형태 데이터 NCHW 1,3,8,8
	cout << "input 4차 배열" << endl;

	for (int c = 0; c < FeatureNum; c++)
	{
	for (int y = 0; y < FeatureHeight; y++)
	{
		for (int x = 0; x < FeatureWidth; x++)
		{
			cout << setw(3) << Input[0][c][y][x] << "  ";
		}cout << endl;
	}cout << endl; cout << endl;
	}cout << endl; cout << endl;
	


	//**Handle**
	cudnnHandle_t cudnn;
	cudnnCreate(&cudnn);

	//GPU에 입력행렬 메모리 할당 및 값 복사
	float * dev_Input;
	cudaMalloc((void**)&dev_Input, sizeof(Input) );
	cudaMemcpy(dev_Input, Input, sizeof(Input), cudaMemcpyHostToDevice);

	//입력행렬 구조체 선언, 할당, 초기화
	cudnnTensorDescriptor_t in_desc; //입력 데이터 셋 정보를 갖고 있는 구조체를 가리키기 위한 포인터
	cudnnCreateTensorDescriptor(&in_desc); // 4D tensor 구조체 객체 생성
	cudnnSetTensor4dDescriptor( // 4D tensor 구조체 초기화 함수
		/*tensorDesc,*/ in_desc,
		/*format,*/CUDNN_TENSOR_NCHW,
		/*dataType,*/CUDNN_DATA_FLOAT,
		/*Number of images*/ImageNum,
		/*C*/FeatureNum,
		/*H*/FeatureHeight,
		/*W*/FeatureWidth);

	//필터 사이즈 지정 - 추후 조정 가능하도록 KCRS

	//K represents the number of output feature maps, 
	//C the number of input feature maps, == FeatureNum
	//R the number of rows per filter, and 
	//S the number of columns per filter.)

	const int filt_n = 3;
	const int filt_c = 3;
	const int filt_h = 7;
	const int filt_w = 7;

	float Filter[filt_n][filt_c][filt_h][filt_w];

	float sigma = sqrt(2 / (float)(filt_n + filt_c));

	random_device rd;
	mt19937 gen(rd());
	normal_distribution<float> d(0, sigma);


	//Filter 정의
	for (int och = 0; och < filt_n; och++)
	{
		for (int ch = 0; ch < filt_c; ch++)
		{
			for (int row = 0; row < filt_h; row++)
			{
				for (int col = 0; col < filt_w; col++)
				{

					Filter[och][ch][row][col] = (1 + och)*0.01;
					//Filter[och][ch][row][col] = d(gen);
				}
			}
		}
	}

	cout << "Filter결과" << endl;

	for (int och = 0; och < filt_n; och++)
	{
		for (int ch = 0; ch < filt_c; ch++)
		{
			for (int row = 0; row < filt_h; row++)
			{
				for (int col = 0; col < filt_w; col++)
				{
				cout<<Filter[och][ch][row][col] << "  ";
				}cout << endl;
			}cout << endl; cout << endl;
		}cout << endl; cout << endl;
	}

	//GPU에 필터행렬 복사
	float * dev_Filt;
	cudaMalloc((void**)&dev_Filt, sizeof(float) * filt_n * filt_c * filt_h * filt_w);
	cudaMemcpy(dev_Filt, Filter, sizeof(float) * filt_n * filt_c * filt_h * filt_w, cudaMemcpyHostToDevice);

	//필터구조체 선언, 생성, 초기화
	cudnnFilterDescriptor_t filt_desc; // 필터 정보를 갖는 구조체를 가리키기 위한 포인터
	cudnnCreateFilterDescriptor(&filt_desc); // 필터 구조체 생성 
	cudnnSetFilter4dDescriptor( // 4d filter 구조체 객체 초기화
		/*filterDesc,*/filt_desc,
		/*dataType,*/CUDNN_DATA_FLOAT,
		/*format,*/CUDNN_TENSOR_NCHW,
		/*Number of output feature maps*/filt_n,
		/*Number of input feature maps.*/filt_c,
		/*Height of each filter.*/filt_h,
		/*Width of each filter.*/filt_w);



	//Convolution 연산에서의 값들 지정 - 추후 조정 가능하도록
	const int pad_h = 1; //padding 높이
	const int pad_w = 1; //padding 넓이
	const int str_h = 1; //stride 높이
	const int str_w = 1; //stride 넓이
	const int dil_h = 1; //dilated 높이
	const int dil_w = 1; //dilated 넓이

						 //Convolution 구조체 선언 및 할당
	cudnnConvolutionDescriptor_t conv_desc; // Convolution 수행을 위한 정보를 갖는 구조체 포인터 
	cudnnCreateConvolutionDescriptor(&conv_desc); // Convolution 구조체 객체 생성
	cudnnSetConvolution2dDescriptor(//
		/*convDesc,*/conv_desc,
		/*zero-padding height*/pad_h,
		/*zero-padding width*/pad_w,
		/*Vertical filter stride*/str_h,
		/*Horizontal filter stride*/str_w,
		/*Filter height dilation*/dil_h,
		/*Filter width dilation*/dil_w,
		/*mode*/CUDNN_CONVOLUTION,
		/*computeType*/CUDNN_DATA_FLOAT);


	//Convolution 결과 저장행렬 선언 및 할당
	cudnnTensorDescriptor_t out_conv_desc;
	cudnnCreateTensorDescriptor(&out_conv_desc);

	//Convolution 결과행렬 사이즈 도출
	int out_conv_n;
	int out_conv_c;
	int out_conv_h;
	int out_conv_w;

	cudnnGetConvolution2dForwardOutputDim( // 주어진 필터, tensor, convolution 구조체 정보를 바탕으로, 2D convolution 계산에 의한 4d tensor의 결과 값의 차원을 반환, 즉 계산 output의 차원 수 인듯...  
		/*convolution descriptor*/conv_desc,
		/*tensor descriptor*/in_desc,
		/*filter descriptor*/filt_desc,
		/*Output. Number of output images*/&out_conv_n,
		/*Output. Number of output feature maps per image.*/&out_conv_c,
		/*Output. Height of each output feature map.*/&out_conv_h,
		/*Output. Width of each output feature map.*/&out_conv_w);

	//outputDim = 1 + ( inputDim + 2*pad - (((filterDim-1)*dilation)+1) )/convolutionStride

	int outputDim = 1 + (FeatureHeight + 2 * pad_h - filt_h) / str_h;
	const int outputDimHW = 4;

	//Convolution행렬 선언
	float Output_Conv[ImageNum][FeatureNum][outputDimHW][outputDimHW];


	//GPU에 Convolution 결과 행렬 할당
	float * dev_Output_Conv;
	cudaMalloc((void**)&dev_Output_Conv, sizeof(float) * out_conv_c * out_conv_h * out_conv_n * out_conv_w);

	//Convolution 구조체 초기화
	cudnnSetTensor4dDescriptor(out_conv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
		ImageNum, FeatureNum, outputDim, outputDim);

	//입력과 필터, 컨볼루션 패딩, 스트라이드가 위와 같이 주어졌을때 가장 빠른 알고리즘이 무엇인지를 알아내기
	cudnnConvolutionFwdAlgo_t alg;
	alg = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;

	//Conv 버퍼 데이터크기 알아내는 법 및 버퍼 메모리 할당 추가
	size_t WS_size = 0;
	cudnnGetConvolutionForwardWorkspaceSize(// This function returns the amount of GPU memory workspace
		cudnn, in_desc, filt_desc, conv_desc, out_conv_desc, alg, &WS_size);

	size_t * dev_WS;
	cudaMalloc((void**)&dev_WS, WS_size);

	//연산
	float alpha = 1.0;
	float beta = 0.0;

	checkCUDNN(cudnnConvolutionForward(
		cudnn,
		&alpha,
		in_desc,
		dev_Input,
		filt_desc,
		dev_Filt,
		conv_desc,
		alg,
		dev_WS,
		WS_size,
		&beta,
		out_conv_desc,
		dev_Output_Conv));

	//Convolution결과 GPU로 복사
	cudaMemcpy(Output_Conv, dev_Output_Conv, sizeof(float) * 16*3, cudaMemcpyDeviceToHost);

	cout << "Convolution결과" << endl;
		for (int c = 0; c < FeatureNum; c++)
		{
			for (int y = 0; y < outputDimHW; y++)
			{
				for (int x = 0; x < outputDimHW; x++)
				{
					cout << setw(3) << Output_Conv[0][c][y][x] << "  ";
				}cout << endl;
			}cout << endl; cout << endl;
		}cout << endl; cout << endl;



		//**Bias**

	

		//Bias 결과 저장행렬 선언
		float Output_Bias[ImageNum][FeatureNum][outputDimHW][outputDimHW];

		//bias 값 지정

		float biasValue[filt_n];

		for (int i = 0; i < filt_n; i++) {
			 biasValue[i]= 0.0f;
		}
		for (int i = 0; i < filt_n; i++) {
			cout << biasValue[i] << "  ";
		}

		//GPU에 bias값 복사
		float * dev_Bias;
		cudaMalloc((void**)&dev_Bias, sizeof(biasValue));
		cudaMemcpy(dev_Bias, biasValue, sizeof(biasValue), cudaMemcpyHostToDevice);

		//bias결과 저장행렬 선언, 할당
		cudnnTensorDescriptor_t bias_desc;
		cudnnCreateTensorDescriptor(&bias_desc);
		cudnnSetTensor4dDescriptor( bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, filt_n, 1, 1);

		//bias 덧셈 수행 
		cudnnAddTensor(cudnn, &alpha, bias_desc, dev_Bias,
			&alpha, /*input -> output*/out_conv_desc, /*input -> output*/dev_Output_Conv);

		//Bias합 결과
		cudaMemcpy(Output_Bias, dev_Output_Conv,
			sizeof(float) * ImageNum * FeatureNum * outputDimHW * outputDimHW, cudaMemcpyDeviceToHost);



		//Bias합 결과
	
		std::cout << std::endl << std::endl << "Add Bias (bias : -10)" << std::endl << std::endl;

		for (int c = 0; c < filt_n; c++)
		{
		    for (int i = 0; i < out_conv_h; i++)
		    {
		        for (int j = 0; j < out_conv_w; j++)
		        {
		            std::cout << setw(5) << Output_Bias[0][c][i][j] << "  ";
		        }

		        std::cout << std::endl;
		    } std::cout << std::endl; std::cout << std::endl;
		}



		//Activation Function 구조체 선언 및 할당 
		cudnnActivationDescriptor_t act_desc;
		cudnnCreateActivationDescriptor(&act_desc);

		//Activation Function 종류 지정 - 추후 조정가능하도록
		cudnnActivationMode_t Activation_Function;
		Activation_Function = CUDNN_ACTIVATION_RELU;
		//Activation_Function = CUDNN_ACTIVATION_TANH; 
		//Activation_Function = CUDNN_ACTIVATION_SIGMOID;

		cudnnSetActivationDescriptor(act_desc, Activation_Function, CUDNN_PROPAGATE_NAN, 0);

		//Activation Function 메모리 GPU에 복사
		float * dev_Output_Act;
		cudaMalloc((void**)&dev_Output_Act, sizeof(float) * outputDimHW * outputDimHW * 3);


		//Activatin Function 연산수행
		cudnnActivationForward(
			cudnn, act_desc, &alpha, out_conv_desc, dev_Output_Conv,
			&beta, out_conv_desc, dev_Output_Act);

		//Activation Function 결과값 저장 행렬
		float Output_Activation[ImageNum][FeatureNum][outputDimHW][outputDimHW];
		cudaMemcpy(Output_Activation, dev_Output_Act, sizeof(float) * ImageNum * FeatureNum * outputDimHW * outputDimHW, cudaMemcpyDeviceToHost);

		//Actavation Function 결과
		
		std::cout << std::endl << std::endl << "Activation Function 결과" << std::endl << std::endl;
		for (int c = 0; c < filt_n; c++)
		{
		for (int i = 0; i < outputDimHW; i++)
		{
		for (int j = 0; j < outputDimHW; j++)
		{
		std::cout << setw(5) << Output_Activation[0][c][i][j] << "  ";
		}
		std::cout << std::endl;
		}std::cout << std::endl; std::cout << std::endl;

		}



		//**Pooling연산**


		//Pooling 연산에서 값들 지정 - 추후 선택 가능하도록

		const int pool_wind_h = 2;
		const int pool_wind_w = 2;
		const int pool_pad_h = 0;
		const int pool_pad_w = 0;
		const int pool_strd_w = 2;
		const int pool_strd_h = 2;

		//Pooling 구조체 선언 및 할당 - 추후 Pooling 모드 조정 가능하도록
		cudnnPoolingDescriptor_t pool_desc;
		cudnnCreatePoolingDescriptor(&pool_desc);
		cudnnSetPooling2dDescriptor(pool_desc, CUDNN_POOLING_MAX, CUDNN_PROPAGATE_NAN,
			pool_wind_h, pool_wind_w, pool_pad_h, pool_pad_w, pool_strd_h, pool_strd_w);

		//Pooling 결과저장 행렬 선언 및 할당
		cudnnTensorDescriptor_t out_pool_desc;
		cudnnCreateTensorDescriptor(&out_pool_desc);

		//Pooling 결과행렬 사이즈
		int out_pool_n;
		int out_pool_c;
		int out_pool_h;
		int out_pool_w;

		//Pooling 결과행렬 사이즈 도출
		cudnnGetPooling2dForwardOutputDim(pool_desc, out_conv_desc,
			&out_pool_n, &out_pool_c, &out_pool_h, &out_pool_w);

		//GPU에 Pooling 결과행렬 메모리할당
		float * dev_Output_Pool;
		cudaMalloc((void**)&dev_Output_Pool,
			sizeof(float) * out_pool_n * out_pool_c * out_pool_h * out_pool_w);

		//Pooling 저장행렬 구조체 초기화
		cudnnSetTensor4dDescriptor(out_pool_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
			out_pool_n, out_pool_c, out_pool_h, out_pool_w);

		//Pooling연산 수행
		cudnnPoolingForward(cudnn, pool_desc, &alpha, out_conv_desc, dev_Output_Act,
			&beta, out_pool_desc, dev_Output_Pool);

		//Pooling결과
		float Output_Pool[ImageNum][FeatureNum][(outputDimHW + 2 * pool_pad_h) / pool_strd_h][(outputDimHW + 2 * pool_pad_w) / pool_strd_w];
		cudaMemcpy(Output_Pool, dev_Output_Pool,
			sizeof(float) * out_pool_n * out_pool_c * out_pool_h * out_pool_w, cudaMemcpyDeviceToHost);

		//Pooling행렬 
		
		std::cout << std::endl << std::endl << "Pooling 결과" << std::endl << std::endl;

		for (int c = 0; c < filt_n; c++)
		{
		for (int i = 0; i < out_pool_h; i++)
		{
		for (int j = 0; j < out_pool_w; j++)
		{
		std::cout << setw(5) << Output_Pool[0][c][i][j] << "  ";
		}
		std::cout << std::endl; std::cout << std::endl;
		}
		}	std::cout << std::endl;	std::cout << std::endl;





		//**Fully Connected**
		//*******************

		//Weights 선언
		float Weights[10][3][2][2];

		//Weights 정의
		for (int och = 0; och < 10; och++)
		{
			for (int ch = 0; ch < 3; ch++)
			{
				for (int row = 0; row < 2; row++)
				{
					for (int col = 0; col < 2; col++)
					{
						Weights[och][ch][row][col] = 0.1 + och*0.1;
					}
				}
			}
		}



		//GPU에 Weights행렬 복사
		float * dev_weights;
		cudaMalloc((void**)&dev_weights, sizeof(float) * 10 * 3 * 2 * 2);
		cudaMemcpy(dev_weights, Weights, sizeof(float) * 10 * 3 * 2 * 2, cudaMemcpyHostToDevice);

		//Weights를 위한 Filter 구조체 선언 및 할당
		cudnnFilterDescriptor_t weights_desc;
		cudnnCreateFilterDescriptor(&weights_desc);
		cudnnSetFilter4dDescriptor(weights_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 10, 3, 2, 2);

		//Fully Connected를 위한 Convolution 구조체 선언 및 할당
		cudnnConvolutionDescriptor_t fc_desc;
		cudnnCreateConvolutionDescriptor(&fc_desc);
		cudnnSetConvolution2dDescriptor(fc_desc, 0, 0, 1, 1, 1, 1, CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT);

		//Fully Connected 연산 결과 저장행렬 구조체 선언
		cudnnTensorDescriptor_t out_fc_desc;
		cudnnCreateTensorDescriptor(&out_fc_desc);

		//Fully Connected 결과행렬 사이즈 도출
		int out_fc_n;
		int out_fc_c;
		int out_fc_h;
		int out_fc_w;

		cudnnGetConvolution2dForwardOutputDim(
			fc_desc, out_fc_desc, weights_desc, &out_fc_n, &out_fc_c, &out_fc_h, &out_fc_w);

		//FC 결과행렬 선언
		float Output_FC[ImageNum][10][1][1];

		//GPU에 FC 결과행렬 할당
		float *dev_Output_FC;
		cudaMalloc((void**)&dev_Output_FC, sizeof(float) * ImageNum * 10 * 1 * 1);

		//FC 구조체 초기화
		cudnnSetTensor4dDescriptor(out_fc_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, ImageNum, 10, 1, 1);


		//FC 버퍼크기 할당 및 도출
		size_t WS_size2 = 0;
		cudnnGetConvolutionForwardWorkspaceSize(
			cudnn, out_pool_desc, weights_desc, fc_desc, out_fc_desc, alg, &WS_size2);

		size_t * dev_WS2;
		cudaMalloc((void**)&dev_WS2, WS_size2);

		//Fully Connected 연산 
		cudnnConvolutionForward(
			cudnn, &alpha, out_pool_desc, dev_Output_Pool, weights_desc, dev_weights, fc_desc,
			alg, dev_WS2, WS_size2, &beta, out_fc_desc, dev_Output_FC);

		//FC 결과를 CPU에 저장
		cudaMemcpy(Output_FC, dev_Output_FC, sizeof(float) * ImageNum * 10 * 1 * 1, cudaMemcpyDeviceToHost);


		
		std::cout << std::endl << std::endl << "Weights" << std::endl << std::endl;

		for (int och = 0; och < 10; och++)
		{
		    for (int ch = 0; ch < 3; ch++)
		    {
		        for (int row = 0; row < 2; row++)
		        {
		            for (int col = 0; col < 2; col++)
		            {
		                std::cout << setw(5) << Weights[och][ch][row][col] << "  ";
		            } std::cout << std::endl;
		        }std::cout << std::endl; 
		    }std::cout << std::endl; std::cout << std::endl;
		}


		std::cout << std::endl << std::endl << "Fully Connected Vector" << std::endl << std::endl;

		for (int i = 0; i < 10; i++)
		{
		std::cout << setw(5) << Output_FC[0][i][0][0] << "  ";
		}

		std::cout << std::endl;
		

		//FC bias 결과 저장
		float Output_FC_Bias[ImageNum][10][1][1];

		//FC bias값
		float biasValueFC[10];


		for (int i = 0; i < 10; i++) {
			biasValueFC[i] = -5.0f;
		}
		for (int i = 0; i < 10; i++) {
			cout << biasValueFC[i] << "  ";
		}



		//GPU에 FC bias값 복사
		float * dev_Bias_FC;
		cudaMalloc((void**)&dev_Bias_FC, sizeof(float)*10);
		cudaMemcpy(dev_Bias_FC, biasValueFC, sizeof(float)* 10, cudaMemcpyHostToDevice);


		//FC Softmax 구조체 - 얘 왜 여기있지?
		cudnnTensorDescriptor_t out_Bias_FC_desc;
		cudnnCreateTensorDescriptor(&out_Bias_FC_desc);
		cudnnSetTensor4dDescriptor(out_Bias_FC_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 10, 1, 1);


		//bias 덧셈 수행
		cudnnAddTensor(cudnn, &alpha, out_Bias_FC_desc, dev_Bias_FC, &alpha, out_fc_desc, dev_Output_FC);
		
		cudaMemcpy(Output_FC_Bias, dev_Output_FC, sizeof(float) * ImageNum * 10, cudaMemcpyDeviceToHost);



		std::cout << std::endl << std::endl << "bias Vector" << std::endl << std::endl;

		for (int i = 0; i < 10; i++)
		{
			std::cout << setw(5) << Output_FC_Bias[0][i][0][0] << "  ";
			//Output_FC_Bias[0][i][0][0] = Output_FC_Bias[0][i][0][0] / 100000.0 ;

		}

		std::cout << std::endl;

		float * dev_Output_FC2;
		cudaMalloc((void**)&dev_Output_FC2, sizeof(float) * ImageNum * 10);
		cudaMemcpy(dev_Output_FC2, Output_FC_Bias, sizeof(float) * ImageNum * 10, cudaMemcpyHostToDevice);




		float OutSoft[ImageNum][10][1][1];

		float * dev_Output_Softmax;
		cudaMalloc((void**)&dev_Output_Softmax, sizeof(float) * ImageNum * 10);


		cudnnSoftmaxForward(cudnn, CUDNN_SOFTMAX_FAST, CUDNN_SOFTMAX_MODE_INSTANCE/*이 부분 매우 중요 - INSTANCE로 해야 바로 미분계산*/,
			&alpha, out_fc_desc, dev_Output_FC2, &beta, out_fc_desc, dev_Output_Softmax);

		cudaMemcpy(OutSoft, dev_Output_Softmax, sizeof(float) * ImageNum * 10, cudaMemcpyDeviceToHost);

		std::cout << std::endl << std::endl << "OutSoft Vector" << std::endl << std::endl;

		for (int n = 0; n < ImageNum; n++)
		{
			for (int c = 0; c < 10; c++) {

			cout<<	OutSoft[n][c][0][0] << "  "  ;
			}
		}std::cout << std::endl;


		//======================
		for (int n = 0; n < ImageNum; n++)
		{
			for (int c = 0; c < 10; c++) {

				OutSoft[n][c][0][0]*=(-1.0);
			}
		}

		 OutSoft[0][0][0][0] = 1.0 + OutSoft[0][0][0][0] ;


		 std::cout << std::endl << std::endl << "dev_dloss Vector" << std::endl << std::endl;

		 for (int n = 0; n < ImageNum; n++)
		 {
			 for (int c = 0; c < 10; c++) {

				 cout << OutSoft[n][c][0][0] << "  ";
			 }
		 }std::cout << std::endl;





		 float * dev_dloss;
		 cudaMalloc((void**)&dev_dloss, sizeof(float) * 10);
		 cudaMemcpy(dev_dloss, OutSoft, sizeof(float) * 10, cudaMemcpyHostToDevice);


	
		 //*********************************
		 //**Fully Connected Bias Backward**
		 //*********************************



		 //저장행렬
		 float FCbiasBack[ImageNum][10][1][1];

		 //GPU 메모리
		 float * dev_FC_bias_Back;
		 cudaMalloc((void**)&dev_FC_bias_Back, sizeof(float)*10);

		 cudnnConvolutionBackwardBias(cudnn, &alpha, out_fc_desc, dev_dloss, &beta, out_Bias_FC_desc, dev_FC_bias_Back);

		 cudaMemcpy(FCbiasBack, dev_FC_bias_Back, sizeof(float)*10, cudaMemcpyDeviceToHost);


		 std::cout << std::endl << std::endl << "FCbiasdelta Vector" << std::endl << std::endl;
		 for (int i = 0; i < 10; i++) {

			 std::cout << FCbiasBack[0][i][0][0] << "  " ;
		 }std::cout << std::endl;
		 
		 std::cout << std::endl << std::endl << "updated fc_bias Vector" << std::endl << std::endl;
		 for (int i = 0; i < 10; i++) {
			 biasValueFC[i] = FCbiasBack[0][i][0][0];
		 }



		 //**********************************
		 //**Fully Connected Backpropagtion**
		 //**********************************

		 //저장행렬 선언
		 float FC_delta_Weights[10][3][2][2];

		 //GPU에 메모리 할당
		 float * dev_Filter_Gradient;
		 cudaMalloc((void**)&dev_Filter_Gradient, sizeof(float) * 3 * 10 * 2 * 2);

		 // Workspace
		 size_t WS_size3 = 0;
		 cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn, weights_desc, out_fc_desc, fc_desc, out_pool_desc, CUDNN_CONVOLUTION_BWD_DATA_ALGO_0, &WS_size3);

		 //GPU에 workspace 메모리 할당
		 size_t * dev_WS3;
		 cudaMalloc((void**)&dev_WS3, WS_size3);

		 //Fully Connected Backpropagation delta
		 cudnnConvolutionBackwardFilter(cudnn, &alpha,
			 out_pool_desc, dev_Output_Pool, out_fc_desc, dev_dloss, fc_desc, CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,
			 dev_WS3, WS_size3, &beta, weights_desc, dev_Filter_Gradient);

		 //CPU에 결과 복사
		 cudaMemcpy(FC_delta_Weights, dev_Filter_Gradient, sizeof(float) * 3 * 10 * 2 * 2, cudaMemcpyDeviceToHost);

		 std::cout << std::endl << std::endl << "FC_delta_Weights" << std::endl << std::endl;
		
		 //weight update
		 float learning_rate = -0.01f;
		
		 for (int och = 0; och < 10; och++)
		 {
			 for (int ch = 0; ch < 3; ch++)
			 {
				 for (int row = 0; row < 2; row++)
				 {
					 for (int col = 0; col < 2; col++)
					 {
						 std::cout << setw(5) << FC_delta_Weights[och][ch][row][col] << "  ";

						 Weights[och][ch][row][col] += FC_delta_Weights[och][ch][row][col] * learning_rate;

					 } std::cout << std::endl;
				 }std::cout << std::endl; 
			 }std::cout << std::endl; std::cout << std::endl;
		 }

	

		 std::cout << std::endl << std::endl << "updated Weights" << std::endl << std::endl;

		 for (int och = 0; och < 10; och++)
		 {
			 for (int ch = 0; ch < 3; ch++)
			 {
				 for (int row = 0; row < 2; row++)
				 {
					 for (int col = 0; col < 2; col++)
					 {
						 std::cout << setw(5) << Weights[och][ch][row][col] << "  ";
					 } std::cout << std::endl;
				 }std::cout << std::endl; std::cout << std::endl;
			 }std::cout << std::endl; std::cout << std::endl;
		 }



		 //**********************************
		 //**fc 바로전, pooling 결과값 delta Backpropagtion**
		 //**********************************


		 //저장행렬 선언
		 float FC_delta_data[ImageNum][3][2][2]; // fc 바로 전 값 

		 float * dev_data_Gradient;
		 cudaMalloc((void**)&dev_data_Gradient, sizeof(float) * ImageNum * 3 * 2 * 2);

		 cudnnConvolutionBackwardData(cudnn,
			 &alpha,
			 weights_desc, dev_weights,
			 out_fc_desc, dev_dloss,
			 fc_desc, CUDNN_CONVOLUTION_BWD_DATA_ALGO_0,
			 dev_WS3, WS_size3,
			 &beta,
			 out_pool_desc, dev_data_Gradient);
		 //CPU에 결과 복사
		 cudaMemcpy(FC_delta_data, dev_data_Gradient, sizeof(float) * 3 * ImageNum * 2 * 2, cudaMemcpyDeviceToHost);


		 
		 std::cout << std::endl << std::endl << "FC_delta_data" << std::endl << std::endl;

		 for (int n = 0; n < ImageNum; n++)
		 {
		 for (int c = 0; c < 3; c++)
		 {
		 for (int i = 0; i < 2; i++)
		 {
		 for (int j = 0; j < 2; j++)
		 {
		 std::cout << setw(3) << FC_delta_data[n][c][i][j] << " :: ";
		 }
		 std::cout << std::endl;
		 }
		 std::cout << std::endl;
		 }
		 std::cout << "==========================================" << endl;
		 std::cout << std::endl; std::cout << std::endl;
		 }
		 

		 //***************************
		 //**Pooling Backpropagation**
		 //***************************

		 //Pooling Backward 저장행렬 선언 및 GPU에 메모리 할당
		 float PoolingBack[ImageNum][3][4][4];


		 float * dev_Pool_Back;
		 cudaMalloc((void**)&dev_Pool_Back, sizeof(float) *ImageNum * 3 * 4 * 4);


		 //구조체 선언부터 초기화
		 cudnnTensorDescriptor_t pool_back_desc;
		 cudnnCreateTensorDescriptor(&pool_back_desc);
		 cudnnSetTensor4dDescriptor(pool_back_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, ImageNum, 3, 4, 4);


		 cudnnPoolingBackward(cudnn, pool_desc, &alpha,
			 out_pool_desc, dev_Output_Pool,
			 /**/out_pool_desc, dev_data_Gradient/**/,
			 out_conv_desc, dev_Output_Act,
			 &beta, pool_back_desc, dev_Pool_Back);

		 cudaMemcpy(PoolingBack, dev_Pool_Back, sizeof(float) *ImageNum * 3 * 4 * 4, cudaMemcpyDeviceToHost);
		
		 std::cout << std::endl << std::endl << "PoolingBack" << std::endl << std::endl;

		 for (int n = 0; n < ImageNum; n++)
		 {
		 for (int c = 0; c < 3; c++)
		 {
		 for (int i = 0; i < 4; i++)
		 {
		 for (int j = 0; j < 4; j++)
		 {
		 std::cout << setw(8) << PoolingBack[n][c][i][j] << " :: ";
		 }
		 std::cout << std::endl;
		 }
		 std::cout << std::endl;
		 }
		 std::cout << "==========================================" << endl;
		 std::cout << std::endl; std::cout << std::endl;
		 }


		 //******************************
		 //**Activation Backpropagation**
		 //******************************
		 


		 float ActBack[ImageNum][3][4][4];
		 float * dev_Act_Back;
		 cudaMalloc((void**)&dev_Act_Back, sizeof(float) *ImageNum * 3 * 4 * 4);

		 cudnnActivationBackward(cudnn, act_desc, &alpha,
			 out_conv_desc, dev_Output_Act,
			 out_conv_desc, dev_Pool_Back,
			 out_conv_desc, dev_Output_Conv,
			 &beta, out_conv_desc, dev_Act_Back);

		 cudaMemcpy(ActBack, dev_Act_Back, sizeof(float) *ImageNum * 3 * 4 * 4, cudaMemcpyDeviceToHost);

		 
		 std::cout << std::endl << std::endl << "ActBack" << std::endl << std::endl;

		 for (int n = 0; n < ImageNum; n++)
		 {
			 for (int c = 0; c < 3; c++)
			 {
				 for (int i = 0; i < 4; i++)
				 {
					 for (int j = 0; j < 4; j++)
					 {
						 std::cout << setw(8) << ActBack[n][c][i][j] << " :: ";
					 }

					 std::cout << std::endl;
				 }

				 std::cout << std::endl;
			 }

			 std::cout << "==========================================" << endl;
			 std::cout << std::endl; std::cout << std::endl;
		 }



		 //*********************************
		 //**Conv Bias Backward**
		 //*********************************   Output_Conv_Bias

		 //저장행렬
		 float ConvbiasBack[3];

		 //GPU 메모리
		 float * dev_Conv_bias_Back;
		 cudaMalloc((void**)&dev_Conv_bias_Back, sizeof(float) * 3);
		 cudnnConvolutionBackwardBias(cudnn, &alpha,
			 out_conv_desc, dev_Act_Back, // dy
			 &beta,
			 bias_desc, dev_Conv_bias_Back); // db


		 cudaMemcpy(ConvbiasBack, dev_Conv_bias_Back, sizeof(float)*3, cudaMemcpyDeviceToHost);

		 for (int i = 0; i < 3; i++) {
			 cout<< ConvbiasBack[i] << " ";
		 }cout << endl;


		 //******************************
		 //**Conv filter Backpropagation**
		 //******************************

		 //dev_Filter_conv


		 float Filter_conv[3][3][7][7];

		 float * dev_Filter_conv2;
		 cudaMalloc((void**)&dev_Filter_conv2, sizeof(float) * 3 * 3 * 7 * 7);

		 // Workspace
		 size_t WS_size4 = 0;
		 cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn, filt_desc, out_conv_desc,
			 fc_desc, in_desc, CUDNN_CONVOLUTION_BWD_DATA_ALGO_0, &WS_size4);

		 //GPU에 workspace 메모리 할당
		 size_t * dev_WS4;
		 cudaMalloc((void**)&dev_WS4, WS_size4);

		 //Fully Connected Backpropagation delta
		 cudnnConvolutionBackwardFilter(cudnn, &alpha,
			 in_desc, dev_Input, //  x 입력값				  // ImageNum, 3, 32, 32
			 out_conv_desc, dev_Act_Back, // dy 바로전 act_dev // ImageNum, 3, 8, 8
			 conv_desc, CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,
			 dev_WS4, WS_size4, &beta,
			 filt_desc, dev_Filter_conv2); // dw 필터			// 3 ,3, 8, 8

		 cudaMemcpy(Filter_conv, dev_Filter_conv2, sizeof(float) * 3 * 3 * 7 * 7, cudaMemcpyDeviceToHost);


		 std::cout << std::endl << std::endl << "Filter_conv" << std::endl << std::endl;

		 for (int n = 0; n < 3; n++)
		 {
			 for (int c = 0; c < 3; c++)
			 {
				 for (int i = 0; i < 7; i++)
				 {
					 for (int j = 0; j < 7; j++)
					 {
						 std::cout << setw(8) << Filter_conv[n][c][i][j] << " :: ";
					 }
					 std::cout << std::endl;
				 }
				 std::cout << std::endl;
			 }
			 std::cout << "==========================================" << endl;
			 std::cout << std::endl; std::cout << std::endl;
		 }

		 
		 

}