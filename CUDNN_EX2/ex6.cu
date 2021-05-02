#include <iomanip>
#include <iostream>
#include <cstdlib>
#include <math.h>
#include <cuda.h>
#include <cudnn.h>
#include <stdio.h>
#include <string.h>
#include <opencv2/opencv.hpp>

using namespace cv;
//***********************************************
//**추후에 각 과정마다 GPU 메모리 Size확인 필요**
//***********************************************

//***********************
//**옵션 설정 기능 추가**
//***********************

//********************
//**alpha, beta 주의**
//********************
using namespace std;

int main()
{
	//**********
	//**Handle**
	//**********
	cudnnHandle_t cudnn;
	cudnnCreate(&cudnn);

	//********
	//**변수**
	//********

	//입력변수
	const int ImageNum = 1;
	const int FeatureNum = 3;
	const int FeatureHeight = 32;
	const int FeatureWidth = 32;

	//********
	//**입력**
	//********
	Mat img = imread("D:\\DataSet\\cifar\\test\\0_cat.png");	// 이미지파일을 읽어 들여 Mat 형식으로 저장시키기
	unsigned char* imgd = img.data;
	//입력행렬 선언
	float Input[ImageNum][FeatureNum][FeatureHeight][FeatureWidth];

	//입력행렬 정의
	for (int i = 0; i < 3; i++)
	{
		for (int y = 0; y < 32; y++)
		{
			for (int x = 0; x < 32; x++)
			{
				Input[0][i][y][x] = imgd[3 * 32  * y + 3 * x + i];
			}
		}
	}

	//GPU에 입력행렬 메모리 할당 및 값 복사
	float * dev_Input;
	cudaMalloc((void**)&dev_Input, sizeof(Input));
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

	//************************
	//************************
	//**Feedforward 연산수행**
	//************************
	//************************

	//*******************
	//**Convolution연산**
	//*******************

	//필터 사이즈 지정 - 추후 조정 가능하도록
	const int filt_n = 1;
	const int filt_c = 3;
	const int filt_h = 8;
	const int filt_w = 8;


	//필터 선언
	float Filter[filt_n][filt_c][filt_h][filt_w];

	//필터 정의
	for (int ch = 0; ch < filt_c; ch++)
	{
	for (int row = 0; row < filt_h; row++)
	{
		for (int col = 0; col < filt_w; col++)
		{
			Filter[0][ch][col][row] = (float)((col + row) % 3);
		}

	}
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
	const int pad_h = 2; //padding 높이
	const int pad_w = 2; //padding 넓이
	const int str_h = 4; //stride 높이
	const int str_w = 4; //stride 넓이
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

	//Convolution행렬 선언
	float Output_Conv[ImageNum][FeatureNum][8][8];


	//GPU에 Convolution 결과 행렬 할당
	float * dev_Output_Conv;
	cudaMalloc((void**)&dev_Output_Conv, sizeof(float) * out_conv_c * out_conv_h * out_conv_n * out_conv_w);

	//Convolution 구조체 초기화
	cudnnSetTensor4dDescriptor(out_conv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
		ImageNum, FeatureNum, outputDim, outputDim);

	//입력과 필터, 컨볼루션 패딩, 스트라이드가 위와 같이 주어졌을때 가장 빠른 알고리즘이 무엇인지를 알아내기
	cudnnConvolutionFwdAlgo_t alg;
	cudnnGetConvolutionForwardAlgorithm(
		cudnn,
		in_desc,
		filt_desc,
		conv_desc,
		out_conv_desc,
		CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
		0,
		&alg);
	cout << "Fastest algorithm for conv0 = " << alg << endl;

	//Conv 버퍼 데이터크기 알아내는 법 및 버퍼 메모리 할당 추가
	size_t WS_size = 0;
	cudnnGetConvolutionForwardWorkspaceSize(// This function returns the amount of GPU memory workspace
		cudnn, in_desc, filt_desc, conv_desc, out_conv_desc, alg, &WS_size);

	size_t * dev_WS;
	cudaMalloc((void**)&dev_WS, WS_size);

	//연산
	float alpha = 1.0;
	float beta = 0.0;

	cudnnConvolutionForward(
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
		dev_Output_Conv);

	//Convolution결과 GPU로 복사
	cudaMemcpy(Output_Conv, dev_Output_Conv,
		sizeof(float) * out_conv_n * out_conv_c * out_conv_h * out_conv_w, cudaMemcpyDeviceToHost);

	
	//********
	//**Bias**
	//********
	beta = 1.0f;

	//Bias 결과 저장행렬 선언
	float Output_Bias[ImageNum][FeatureNum][FeatureHeight][FeatureWidth];

	//bias 값 지정
	float biasValue[filt_n] = { 10.0f };

	//GPU에 bias값 복사
	float * dev_Bias;
	cudaMalloc((void**)&dev_Bias, sizeof(float));
	cudaMemcpy(dev_Bias, biasValue, sizeof(float), cudaMemcpyHostToDevice);

	//bias결과 저장행렬 선언, 할당
	cudnnTensorDescriptor_t bias_desc;
	cudnnCreateTensorDescriptor(&bias_desc);
	cudnnSetTensor4dDescriptor(
		bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, filt_n, 1, 1);

	//bias 덧셈 수행 
	cudnnAddTensor(cudnn, &alpha, bias_desc, dev_Bias,
		&beta, /*input -> output*/out_conv_desc, /*input -> output*/dev_Output_Conv);

	//Bias합 결과
	cudaMemcpy(Output_Bias, dev_Output_Conv,
		sizeof(float) * ImageNum * FeatureNum * FeatureHeight * FeatureWidth, cudaMemcpyDeviceToHost);

	//***********************
	//**Actiovation Funtion**
	//***********************

	beta = 0.0;

	//Activation Function 구조체 선언 및 할당 
	cudnnActivationDescriptor_t act_desc;
	cudnnCreateActivationDescriptor(&act_desc);

	//Activation Function 종류 지정 - 추후 조정가능하도록
	cudnnActivationMode_t Activation_Function;
	Activation_Function = CUDNN_ACTIVATION_RELU;
	cudnnSetActivationDescriptor(act_desc, Activation_Function, CUDNN_PROPAGATE_NAN, 0);

	//Activation Function 메모리 GPU에 복사
	float * dev_Output_Act;
	cudaMalloc((void**)&dev_Output_Act, sizeof(float) * 32 * 32*3);


	//Activatin Function 연산수행
	cudnnActivationForward(
		cudnn, act_desc, &alpha, out_conv_desc, dev_Output_Conv,
		&beta, out_conv_desc, dev_Output_Act);

	//Activation Function 결과값 저장 행렬
	float Output_Activation[ImageNum][FeatureNum][FeatureHeight][FeatureWidth];
	cudaMemcpy(Output_Activation, dev_Output_Act,
		sizeof(float) * ImageNum * FeatureNum * FeatureHeight * FeatureWidth,
		cudaMemcpyDeviceToHost);

	//***************
	//**Pooling연산**
	//***************

	//Pooling 연산에서 값들 지정 - 추후 선택 가능하도록
	beta = 0.0;

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
	float Output_Pool[ImageNum][FeatureNum][(FeatureHeight + 2 * pool_pad_h) / pool_strd_h][(FeatureWidth + 2 * pool_pad_w) / pool_strd_w];
	cudaMemcpy(Output_Pool, dev_Output_Pool,
		sizeof(float) * out_pool_n * out_pool_c * out_pool_h * out_pool_w, cudaMemcpyDeviceToHost);


	//*******************
	//**Fully Connected**
	//*******************

	//Weights 선언
	float Weights[1][1][16][32];

	//Weights 정의
	for (int row = 0; row < 16; row++)
	{
		for (int col = 0; col < 32; col++)
		{
			Weights[0][0][row][col] = (float)((row + col) % 4) * 0.2;
			//Weights[0][0][row][col + 3] = 0.99f;
		}
	}

	//GPU에 Weights행렬 복사
	float * dev_weights;
	cudaMalloc((void**)&dev_weights,
		sizeof(float) * 1 * 1 * 16 * 32);
	cudaMemcpy(dev_weights, Weights,
		sizeof(float) * 1 * 1 * 16 * 32, cudaMemcpyHostToDevice);

	//Weights를 위한 Filter 구조체 선언 및 할당
	cudnnFilterDescriptor_t weights_desc;
	cudnnCreateFilterDescriptor(&weights_desc);
	cudnnSetFilter4dDescriptor(
		weights_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 1, 1, 16, 32);

	//Fully Connected를 위한 Convolution 구조체 선언 및 할당
	cudnnConvolutionDescriptor_t fc_desc;
	cudnnCreateConvolutionDescriptor(&fc_desc);
	cudnnSetConvolution2dDescriptor(fc_desc, 0, 16, 1, 16, 1, 1, CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT);

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
	float Output_FC[1][1][1][2];

	//GPU에 FC 결과행렬 할당
	float *dev_Output_FC;
	cudaMalloc((void**)&dev_Output_FC, sizeof(float) * 1 * 1 * 1 * 2);

	//FC 구조체 초기화
	cudnnSetTensor4dDescriptor(out_fc_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 1, 2);


	//입력과 필터, 컨볼루션 패딩, 스트라이드가 위와 같이 주어졌을때 가장 빠른 알고리즘이 무엇인지를 알아내기
	cudnnConvolutionFwdAlgo_t alg2;
	cudnnGetConvolutionForwardAlgorithm(
		cudnn,
		out_pool_desc,
		weights_desc,
		fc_desc,
		out_fc_desc,
		CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
		0,
		&alg2);
	cout << "Fastest algorithm for FC = " << alg2 << endl;



	//FC 버퍼크기 할당 및 도출
	size_t WS_size2 = 0;
	cudnnGetConvolutionForwardWorkspaceSize(
		cudnn, out_pool_desc, weights_desc, fc_desc, out_fc_desc, alg2, &WS_size2);

	size_t * dev_WS2;
	cudaMalloc((void**)&dev_WS2, WS_size2);

	//Fully Connected 연산 
	cudnnConvolutionForward(
		cudnn, &alpha, out_pool_desc, dev_Output_Pool, weights_desc, dev_weights, fc_desc,
		alg2, dev_WS2, WS_size2, &beta, out_fc_desc, dev_Output_FC);

	//FC 결과를 CPU에 저장
	cudaMemcpy(Output_FC, dev_Output_FC, sizeof(float) * 1 * 1 * 1 * 2, cudaMemcpyDeviceToHost);


	//*************************
	//**Fully Conncected Bias**
	//*************************
	beta = 1.0f;

	//FC bias 결과 저장
	float Output_FC_Bias[1][1][1][2];

	//FC bias값
	float biasValueFC[1] = { -5.0f };

	//GPU에 FC bias값 복사
	float * dev_Bias_FC;
	cudaMalloc((void**)&dev_Bias_FC, sizeof(float));
	cudaMemcpy(dev_Bias_FC, biasValueFC, sizeof(float), cudaMemcpyHostToDevice);


	//FC Softmax 구조체 - 얘 왜 여기있지?
	cudnnTensorDescriptor_t out_fc_soft_desc;
	cudnnCreateTensorDescriptor(&out_fc_soft_desc);
	cudnnSetTensor4dDescriptor(out_fc_soft_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 1, 1);


	//bias 덧셈 수행
	cudnnAddTensor(cudnn, &alpha, out_fc_soft_desc, dev_Bias_FC, &beta, out_fc_desc, dev_Output_FC);
	cudaMemcpy(Output_FC_Bias, dev_Output_FC, sizeof(float) * 1 * 2, cudaMemcpyDeviceToHost);


	//***********
	//**Softmax**
	//***********
	beta = 0.0;

	float OutSoft[1][1][1][2];
	float * dev_Output_Softmax;
	cudaMalloc((void**)&dev_Output_Softmax, sizeof(float) * 1 * 2);


	cudnnSoftmaxForward(cudnn, CUDNN_SOFTMAX_FAST, CUDNN_SOFTMAX_MODE_INSTANCE/*이 부분 매우 중요 - INSTANCE로 해야 바로 미분계산*/,
		&alpha, out_fc_desc, dev_Output_FC, &beta, out_fc_desc, dev_Output_Softmax);

	cudaMemcpy(OutSoft, dev_Output_Softmax, sizeof(float) * 1 * 2, cudaMemcpyDeviceToHost);

	//*********
	//**Error**
	//*********

	float error = -log(OutSoft[0][0][0][0]);


	//****************************
	//****************************
	//**Backpropagation 연산수행**
	//****************************
	//****************************

	//***************************
	//**Softmax Backpropagation** - p - y
	//***************************


	//저장행렬 선언
	float SoftBack[1][1][1][2];

	//GPU 메모리 할당
	float * dif_Soft_Back;
	cudaMalloc((void**)&dif_Soft_Back, sizeof(float) * 1 * 2);

	//구조체 선언 및 초기화
	cudnnTensorDescriptor_t dif_soft_desc;
	cudnnCreateTensorDescriptor(&dif_soft_desc);
	cudnnSetTensor4dDescriptor(dif_soft_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 1, 2);


	//delta

	float DivSoft[2] = { 0.01f, 0.0f };

	float * dev_Output_Soft_Back;
	cudaMalloc((void**)&dev_Output_Soft_Back, sizeof(float) * 1 * 2);



	float * dev_dif_Softmax;
	cudaMalloc((void**)&dev_dif_Softmax, sizeof(float) * 1 * 2);
	cudaMemcpy(dev_dif_Softmax, DivSoft, sizeof(float) * 1 * 2, cudaMemcpyHostToDevice);


	cudnnTensorDescriptor_t dif_soft_back;
	cudnnCreateTensorDescriptor(&dif_soft_back);
	cudnnSetTensor4dDescriptor(dif_soft_back, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 1, 2);



	cudnnSoftmaxBackward(cudnn, CUDNN_SOFTMAX_ACCURATE, /*****이부분 매우 중요*****/CUDNN_SOFTMAX_MODE_INSTANCE, &alpha,
		dif_soft_back, dev_Output_Softmax, dif_soft_back, dev_dif_Softmax, &beta, dif_soft_back, dev_Output_Soft_Back);

	cudaMemcpy(SoftBack, dev_Output_Soft_Back, sizeof(float) * 1 * 2, cudaMemcpyDeviceToHost);


	//*********************************
	//**Fully Connected Bias Backward**
	//*********************************

	/*

	//저장행렬
	float FCbiasBack[1][1][1][1];

	//GPU 메모리
	float * dev_FC_bias_Back;
	cudaMalloc((void**)&dev_FC_bias_Back, sizeof(float));

	cudnnConvolutionBackwardBias(cudnn, &alpha, out_fc_desc, dev_Output_Softmax, &beta, bias_desc, dev_FC_bias_Back);

	cudaMemcpy(FCbiasBack, dev_FC_bias_Back, sizeof(float), cudaMemcpyDeviceToHost);

	*/

	//**********************************
	//**Fully Connected Backpropagtion**
	//**********************************

	//저장행렬 선언
	float FCBack[1][3][16][32];

	//GPU에 메모리 할당
	float * dev_Filter_Gradient;
	cudaMalloc((void**)&dev_Filter_Gradient, sizeof(float) * 3 * 16 * 32);

	// Workspace
	size_t WS_size3 = 0;
	cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn, weights_desc, dif_soft_back, fc_desc, out_pool_desc, CUDNN_CONVOLUTION_BWD_DATA_ALGO_0, &WS_size3);

	//GPU에 workspace 메모리 할당
	size_t * dev_WS3;
	cudaMalloc((void**)&dev_WS3, WS_size3);

	//Fully Connected Backpropagation delta
	cudnnConvolutionBackwardFilter(cudnn, &alpha,
		out_pool_desc, dev_Output_Pool, dif_soft_back, dev_Output_Soft_Back, fc_desc, CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,
		dev_WS3, WS_size3, &beta, weights_desc, dev_Filter_Gradient);

	//CPU에 결과 복사
	cudaMemcpy(FCBack, dev_Filter_Gradient, sizeof(float) * 3 * 16 * 32, cudaMemcpyDeviceToHost);


	//***************************
	//**Pooling Backpropagation**
	//***************************

	//Pooling Backward 저장행렬 선언 및 GPU에 메모리 할당
	float PoolingBack[1][3][32][32];
	float * dev_Pool_Back;
	cudaMalloc((void**)&dev_Pool_Back, sizeof(float) * 32 * 32 * 3);

	//Filter값 복사해오기
	float cpy_Filter[3][3];

	for (int row = 0; row < 3; row++)
	{
		for (int col = 0; col < 3; col++)
		{
			cpy_Filter[col][row] = FCBack[0][0][col][row];
		}
	}

	//복사한 Filter Gradient를 GPU 메모리에 할당
	float * dev_Filter_cpy;
	cudaMalloc((void**)&dev_Filter_cpy, sizeof(float) * 9);
	cudaMemcpy(dev_Filter_cpy, cpy_Filter, sizeof(float) * 9, cudaMemcpyHostToDevice);

	//구조체 선언부터 초기화
	cudnnTensorDescriptor_t pool_back_desc;
	cudnnCreateTensorDescriptor(&pool_back_desc);
	cudnnSetTensor4dDescriptor(pool_back_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 3, 32, 32);


	cudnnPoolingBackward(cudnn, pool_desc, &alpha,
		out_pool_desc, dev_Output_Pool, /**/out_pool_desc, dev_Filter_cpy/**/, out_conv_desc, dev_Output_Conv,
		&beta, pool_back_desc, dev_Pool_Back);

	cudaMemcpy(PoolingBack, dev_Pool_Back, sizeof(float) * 32 * 32 * 3, cudaMemcpyDeviceToHost);



	//해제
	cudaFree(dev_Filter_cpy);

	//******************************
	//**Activation Backpropagation**
	//******************************


	//나중에 하자...
	/*
	float ActBack[1][1][6][6];
	float * dev_Act_Back;
	cudaMalloc((void**)&dev_Act_Back, sizeof(float) * 6 * 6);

	cudnnActivationBackward(cudnn, act_desc,
	&alpha, out_conv_desc, dev_Output_Act, pool_back_desc, dev_Pool_Back, out_pool_desc, dev_Output_Pool,
	&beta, out_conv_desc, dev_Act_Back);

	cudaMemcpy(ActBack, dev_Act_Back, sizeof(float) * 6 * 6, cudaMemcpyDeviceToHost);
	*/



	//*******************************
	//**Convolution Backpropagation**
	//*******************************

	float BackConv[1][3][16][16];
	float *dev_Back_Conv;
	cudaMalloc((void**)&dev_Back_Conv, sizeof(float) * 1 * 3 * 16 * 16);

	/*
	cudnnTensorDescriptor_t back_fc_filt;
	cudnnCreateTensorDescriptor(&back_fc_filt);
	cudnnSetTensor4dDescriptor(back_fc_filt, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 6, 6);
	*/

	//out_conv_desc, dev_Output_Conv

	//Convolution BackProp 연산
	cudnnConvolutionBackwardFilter(
		cudnn, &alpha, in_desc, dev_Input, pool_back_desc, dev_Pool_Back,
		conv_desc, CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,
		dev_WS2, WS_size2, &beta, filt_desc, dev_Back_Conv);

	cudaMemcpy(BackConv, dev_Back_Conv, sizeof(float) * 1 * 3 * 16 * 16, cudaMemcpyDeviceToHost);


	//********
	//**해제**
	//********

	//구조체
	cudnnDestroyTensorDescriptor(in_desc);
	cudnnDestroyFilterDescriptor(filt_desc);
	cudnnDestroyConvolutionDescriptor(conv_desc);
	cudnnDestroyTensorDescriptor(out_conv_desc);
	cudnnDestroyTensorDescriptor(out_pool_desc);
	cudnnDestroyPoolingDescriptor(pool_desc);
	cudnnDestroyTensorDescriptor(bias_desc);
	cudnnDestroyActivationDescriptor(act_desc);
	cudnnDestroyFilterDescriptor(weights_desc);
	cudnnDestroyTensorDescriptor(out_fc_desc);
	cudnnDestroyTensorDescriptor(dif_soft_desc);
	cudnnDestroyTensorDescriptor(pool_back_desc);



	//GPU 메모리
	cudaFree(dev_Filt);
	cudaFree(dev_Input);
	cudaFree(dev_WS);
	cudaFree(dev_Output_Conv);
	cudaFree(dev_Output_Act);
	cudaFree(dev_Output_Pool);
	cudaFree(dev_Bias);
	cudaFree(dev_weights);
	cudaFree(dev_WS2);
	cudaFree(dev_Output_FC);
	cudaFree(dev_Bias_FC);

	cudaFree(dev_Filter_Gradient);
	cudaFree(dev_Pool_Back);
	cudaFree(dev_Output_Softmax);
	cudaFree(dif_Soft_Back);
	cudaFree(dev_WS3);
	//cudaFree(dev_Act_Back);

	//************
	//**화면출력**
	//************

	//Input

	std::cout << "*******************" << std::endl << "**Input Data 정보**"
		<< std::endl << "*******************" << std::endl << std::endl;


	std::cout << "Input" << std::endl << std::endl;

	for (int i = 0; i < FeatureHeight; i++)
	{
		for (int j = 0; j < FeatureWidth; j++)
		{
			std::cout << setw(3) << Input[0][0][i][j] << "  ";
		}
		std::cout << std::endl;
	}


	//Filter
	std::cout << std::endl << std::endl << "Filter" << std::endl << std::endl;

	for (int i = 0; i < filt_h; i++)
	{
		for (int j = 0; j < filt_w; j++)
		{
			std::cout << setw(3) << Filter[0][0][i][j] << "  ";
		}
		std::cout << std::endl;
	}

	//Fully Connected Weights

	std::cout << std::endl << std::endl << "Output1 Weights" << std::endl << std::endl;

	for (int i = 0; i < 16; i++)
	{
		for (int j = 0; j < 16; j++)
		{
			std::cout << setw(3) << Weights[0][0][i][j] << "  ";
		}
		std::cout << std::endl;
	}

	std::cout << std::endl;

	std::cout << std::endl << std::endl << "Output2 Weights" << std::endl << std::endl;

	for (int i = 0; i < 16; i++)
	{
		for (int j = 0; j < 16; j++)
		{
			std::cout << setw(3) << Weights[0][0][i][j] << "  ";
		}
		std::cout << std::endl;
	}

	std::cout << std::endl;



	//Convolution행렬

	std::cout << std::endl << std::endl << std::endl <<
		"************************" << std::endl << "**Feedforward 결과출력**"
		<< std::endl << "************************";


	std::cout << std::endl << std::endl << "Convolution 결과" << std::endl << std::endl;

	for (int i = 0; i < out_conv_h; i++)
	{
		for (int j = 0; j < out_conv_w; j++)
		{
			std::cout << setw(3) << Output_Conv[0][0][i][j] << "  ";
		}
		std::cout << std::endl;
	}

	


	//Bias합 결과

	std::cout << std::endl << std::endl << "Add Bias (bias : -10)" << std::endl << std::endl;

	for (int i = 0; i < FeatureHeight; i++)
	{
		for (int j = 0; j < FeatureWidth; j++)
		{
			std::cout << setw(3) << Output_Bias[0][0][i][j] << "  ";
		}
		std::cout << std::endl;
	}

	//Actavation Function 결과

	std::cout << std::endl << std::endl << "Activation Function 결과" << std::endl << std::endl;

	for (int i = 0; i < FeatureHeight; i++)
	{
		for (int j = 0; j < FeatureWidth; j++)
		{
			std::cout << setw(3) << Output_Activation[0][0][i][j] << "  ";
		}
		std::cout << std::endl;
	}


	//Pooling행렬 

	std::cout << std::endl << std::endl << "Pooling 결과" << std::endl << std::endl;

	for (int i = 0; i < out_pool_h; i++)
	{
		for (int j = 0; j < out_pool_w; j++)
		{
			std::cout << Output_Pool[0][0][i][j] << "  ";
		}
		std::cout << std::endl;
	}

	//벡터로 변환 

	std::cout << std::endl << std::endl << "Fully Connected Vector" << std::endl << std::endl;

	for (int i = 0; i < out_pool_h; i++)
	{
		for (int j = 0; j < out_pool_w; j++)
		{
			std::cout << Output_Pool[0][0][i][j] << "  ";
		}
		std::cout << std::endl;
	}

	std::cout << std::endl;


	//Fully Connected행렬

	std::cout << std::endl << std::endl << "Fully Connected 연산 결과" << std::endl << std::endl;

	std::cout << Output_FC[0][0][0][0] << "  " << Output_FC[0][0][0][1] << std::endl;


	//Fully Connected Bias행렬

	std::cout << std::endl << std::endl << "Fully Connected Bias" << std::endl << std::endl;

	std::cout << Output_FC_Bias[0][0][0][0] << "  " << Output_FC_Bias[0][0][0][1] << std::endl;


	//Softmax 결과

	std::cout << std::endl << std::endl << "Softmax 결과" << std::endl << std::endl;

	std::cout << OutSoft[0][0][0][0] << "  " << OutSoft[0][0][0][1] << std::endl;

	//원하는 결과 (One-hot Encoding)

	std::cout << std::endl << std::endl << "One-hot Encoding" << std::endl << std::endl;

	std::cout << "1" << "  " << "0" << std::endl;

	//Cross Entropy

	std::cout << std::endl << std::endl << "Cross Entropy 값" << std::endl << std::endl;

	std::cout << error;

	//********************
	//********************
	//**Back-Propagation**
	//********************
	//********************


	std::cout << std::endl << std::endl << std::endl << std::endl <<
		"************************************" << std::endl << "**Backpropagation 결과(delta) 출력**"
		<< std::endl << "************************************" << std::endl << std::endl
		<< "학습률 : 0.01" << std::endl;


	//Softmax BackProp - delta

	std::cout << std::endl << std::endl << "Diff Softmax - delta Softmax (dZ)" << std::endl << std::endl;

	std::cout << SoftBack[0][0][0][0] << "  " << SoftBack[0][0][0][1] << std::endl;




	//BackProp of Fully Connected Bias

	/*

	std::cout << std::endl << std::endl << "FC Bias Backward - delta bias of FC" << std::endl << std::endl;

	std::cout << FCbiasBack[0][0][0][0] << std::endl;

	*/


	//BackProp of Fully Connected 행렬

	std::cout << std::endl << std::endl << "Output1 - delta Weight1 (dW1)" << std::endl << std::endl;

	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			std::cout  << FCBack[0][0][i][j] << "  ";
		}
		std::cout << std::endl;
	}

	std::cout << std::endl << std::endl << std::endl << "Output2 - delta Weight2 (cW2)" << std::endl << std::endl;

	for (int i = 0; i < 3; i++)
	{
		for (int j = 3; j < 6; j++)
		{
			std::cout << FCBack[0][0][i][j] << "  ";
		}
		std::cout << std::endl;
	}

	std::cout << std::endl;


	//BakProp of Pooling

	std::cout << std::endl << std::endl << "Pool Back - delta Pooling Filter" << std::endl << std::endl;

	for (int i = 0; i < FeatureHeight; i++)
	{
		for (int j = 0; j < FeatureWidth; j++)
		{
			std::cout << PoolingBack[0][0][i][j] << "  ";
		}
		std::cout << std::endl;
	}



	//BackProp of Activation Layer

	/*
	std::cout << std::endl << std::endl << "Act Back - delta Activaton Function" << std::endl << std::endl;

	for (int i = 0; i < FeatureHeight; i++)
	{
	for (int j = 0; j < FeatureWidth; j++)
	{
	std::cout << ActBack[0][0][i][j] << "  ";
	}
	std::cout << std::endl;
	}
	*/

	//BackProp of Convolution

	std::cout << std::endl << std::endl << "Convolution Back - delta Filter (dW)" << std::endl << std::endl;

	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			std::cout << BackConv[0][0][i][j] << "  ";
		}
		std::cout << std::endl;
	}



	std::cout << std::endl << std::endl;


	
}
