

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


#define BW 128
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
using namespace cv;



__global__ void SoftmaxLossBackprop(const float *label, int num_labels, int batch_size, float *diff, float *loss)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= batch_size)
		return;

	loss[idx] = label[idx] - diff[idx];
}

vector<pair<Mat, string>> TraverseFilesUsingDFS(const string& folder_path)
{
	_finddata_t file_info;
	string any_file_pattern = folder_path + "\\*";
	intptr_t handle = _findfirst(any_file_pattern.c_str(), &file_info);
	vector<pair<Mat, string>> ImgBox;

	//If folder_path exsist, using any_file_pattern will find at least two files "." and "..",
	//of which "." means current dir and ".." means parent dir
	if (handle == -1)
	{
		cerr << "folder path not exist: " << folder_path << endl;
		exit(-1);
	}

	//iteratively check each file or sub_directory in current folder
	do
	{
		string file_name = file_info.name; //from char array to string

										   //check whtether it is a sub direcotry or a file
		if (file_info.attrib & _A_SUBDIR)
		{
			if (file_name != "." && file_name != "..")
			{
				string sub_folder_path = folder_path + "\\" + file_name;
				TraverseFilesUsingDFS(sub_folder_path);
				cout << "a sub_folder path: " << sub_folder_path << endl;
			}
		}
		else  //cout << "file name: " << file_name << endl;
		{
			size_t npo1 = file_name.find('_') + 1;
			size_t npo2 = file_name.find('.');
			size_t npo3 = npo2 - npo1;
			string newname = file_name.substr(npo1, npo3);
			string sub_folder_path2 = folder_path + "\\" + file_name;
			Mat img = imread(sub_folder_path2);
			ImgBox.push_back({ { img },{ newname } });
		}
	} while (_findnext(handle, &file_info) == 0);

	//
	_findclose(handle);
	return ImgBox;
}

static inline unsigned int RoundUp(unsigned int nominator, unsigned int denominator)
{
	return (nominator + denominator - 1) / denominator;
}


void InitWeightsXavier(const size_t PrevLayerNodeNumber, const size_t NextLayerNodeNumber, const size_t TotalNumber, vector<float> &Weights)
{

	float sigma = sqrt(2 / (float)(PrevLayerNodeNumber + NextLayerNodeNumber));

	random_device rd;
	mt19937 gen(rd());
	normal_distribution<float> d(0, sigma);

	for (size_t iter = 0; iter < TotalNumber; iter++)
	{

		Weights[iter] = d(gen);
	}
}

void InitWeightsbias(const size_t & numOutSize, vector<float> & Weightsbias)
{
	for (int i = 0; i < numOutSize; i++)
	{
		Weightsbias[i] = 1.0f;
	}
}


int main()
{
	const int numImgs = 100; // 이미지 총 갯수
	string folder_path = "D:\\DataSet\\cifar\\test"; // 이미지가 저장되어 있는 폴더 경로
	vector<pair<Mat, string>> ImgBox; // 이미지 데이터, 이미지 이름
	ImgBox = TraverseFilesUsingDFS(folder_path);
	vector<string> LabelBox; // 라벨 정리를 위해 생성
	vector<pair<int, string>> LabelTable; // 라벨링 마다 넘버 부여
	vector<pair<Mat, int>> ImgBox2; // 이미지 데이터, 라벨 넘버
	vector<float> labels(numImgs);
	vector<vector<int>> TargetY; // 라벨 넘버 -> 이진수 형태 데이터로 저장



								 //필터 선언
	float Filter[3][3][8][8];

	float sigma = sqrt(2 / (float)(3 + 3));

	random_device rd;
	mt19937 gen(rd());
	normal_distribution<float> d(0, sigma);


	//Weights 정의
	for (int och = 0; och < 3; och++)
	{
		for (int ch = 0; ch < 3; ch++)
		{
			for (int row = 0; row < 8; row++)
			{
				for (int col = 0; col < 8; col++)
				{
					Filter[och][ch][row][col] = d(gen);
				}
			}
		}
	}


	float Weights[10][3][4][4];

	float sigma2 = sqrt(2 / (float)(10 + 3));

	random_device rd2;
	mt19937 gen2(rd2());
	normal_distribution<float> d2(0, sigma2);


	//Weights 정의
	for (int och = 0; och < 10; och++)
	{
		for (int ch = 0; ch < 3; ch++)
		{
			for (int row = 0; row < 4; row++)
			{
				for (int col = 0; col < 4; col++)
				{
					Weights[och][ch][row][col] = d2(gen2);
				}
			}
		}
	}


	// 라벨에 번호 부여를 위해 LabelBox 벡터에 값 복사 하고 정렬 및 중복 삭제
	for (int i = 0; i < numImgs; i++)
	{
		LabelBox.push_back(ImgBox[i].second);
	}

	sort(LabelBox.begin(), LabelBox.end());
	LabelBox.erase(unique(LabelBox.begin(), LabelBox.end()), LabelBox.end());
	size_t nLabelBoxSize = LabelBox.size();

	// 라벨 번호 부여
	for (int i = 0; i < nLabelBoxSize; i++)
	{
		LabelTable.push_back({ { i },{ LabelBox[i] } });
	}

	//ImgBox2 셋팅
	for (int i = 0; i < numImgs; i++)
	{
		ImgBox2.push_back({ ImgBox[i].first, 0 });

		for (int j = 0; j < LabelTable.size(); j++)
		{
			if (ImgBox[i].second == LabelTable[j].second)
			{
				ImgBox2[i].second = LabelTable[j].first;
				labels[i] = LabelTable[j].first;
			}
		}
	}


	// TargetY 셋팅, 정답 이진수 형태로 표현
	TargetY.resize(numImgs);

	for (int i = 0; i < numImgs; i++)
	{
		TargetY[i].resize(nLabelBoxSize, 0);
	}

	for (int i = 0; i < numImgs; i++)
	{
		int idx = ImgBox2[i].second;
		TargetY[i][idx] = 1;
	}


	float  *d_labels;
	cudaMalloc((void**)&d_labels, sizeof(float) * numImgs * 10 * 1 * 1);

	cudaMemcpyAsync(d_labels, &TargetY, sizeof(float) * numImgs, cudaMemcpyHostToDevice);




	// 4차 행렬 동적 할당 선언.
	float**** Input = new float** *[numImgs];

	for (int i = 0; i < numImgs; i++)
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

	// mat 형식 - > 4차 행렬
	for (int i = 0; i < numImgs; i++)
	{
		unsigned char* temp = ImgBox2[i].first.data;

		for (int c = 0; c < 3; c++)
		{
			for (int y = 0; y < 32; y++)
			{
				for (int x = 0; x < 32; x++)
				{
					//Input[i][c][y][x] = temp[3 * 32 * y + 3 * x + c];
					Input[i][c][y][x] = temp[3 * 32 * y + 3 * x + c] * (1.0 / 255);
				}
			}
		}
	}



	for (int y = 0; y < 32; y++)
	{
		for (int x = 0; x < 32; x++)
		{
			cout << setw(10) << Input[0][0][y][x] << "::";

		}cout << endl;
	}cout << endl; cout << endl;






	// 이미지 데이터 준비 끝

	//**********
	//**Handle**
	//**********
	cudnnHandle_t cudnn;
	cudnnCreate(&cudnn);
	cublasHandle_t cublasHandle;
	cublasCreate(&cublasHandle);

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
	//GPU에 입력행렬 메모리 할당 및 값 복사

	float* dev_Input;
	cudaMalloc((void**)&dev_Input, sizeof(Input));
	cudaMemcpy(dev_Input, Input, sizeof(Input), cudaMemcpyHostToDevice);//입력행렬 구조체 선언, 할당, 초기화
	cudnnTensorDescriptor_t in_desc; //입력 데이터 셋 정보를 갖고 있는 구조체를 가리키기 위한 포인터
	cudnnCreateTensorDescriptor(&in_desc); // 4D tensor 구조체 객체 생성

										   // 4D tensor 구조체 초기화 함수
	cudnnSetTensor4dDescriptor(
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
	const int filt_n = 3;
	const int filt_c = 3;
	const int filt_h = 8;
	const int filt_w = 8;
	/*
	//필터 선언
	float Filter[filt_n][filt_c][filt_h][filt_w];

	//필터 정의
	int miner = -1;

	for (int och = 0; och < filt_n; och++)
	{
	for (int ch = 0; ch < filt_c; ch++)
	{
	for (int row = 0; row < filt_h; row++)
	{
	for (int col = 0; col < filt_w; col++)
	{
	Filter[och][ch][col][row] = (float)((col + row) % 3) * miner;
	miner *= -1;
	}
	}
	}
	}
	*/
	//std::vector<float> Filter; //[filterCnt][input_channelCnt][convFilterHeight][convFilterWidth]
	//Filter.resize(filt_n * filt_c * filt_h * filt_w); //3, 3, 8, 8, in_channels_ * kernel_size_ * kernel_size_ * out_channels_
	//InitWeightsXavier(filt_c, filt_n, filt_n * filt_c * filt_h * filt_w, Filter);
	/*
	for (int ch = 0; ch < 3*3*8*8; ch++)
	{
	if (ch % 8 == 0)
	cout << endl;
	cout << setw(10) <<Filter[ch] << "::" ;

	}

	*/


	//++++++++++++++++++++++++++++++++++++++++++++++++++++++++

	//GPU에 필터행렬 복사
	float *dev_Filt;
	cudaMalloc((void**)&dev_Filt, sizeof(float) * filt_n * filt_c * filt_h * filt_w);
	cudaMemcpy(dev_Filt, &Filter[0], sizeof(float) * filt_n * filt_c * filt_h * filt_w, cudaMemcpyHostToDevice);

	float Filter_conv[3][3][8][8];

	cudaMemcpyAsync(Filter_conv, dev_Filt, sizeof(float) * 3 * 3 * 8 * 8, cudaMemcpyDeviceToHost);

	/*

	for (int ch = 0; ch < filt_c; ch++)
	{
	for (int row = 0; row < filt_h; row++)
	{
	for (int col = 0; col < filt_w; col++)
	{
	cout << setw(10) << Filter_conv[0][ch][col][row] << "::";
	}cout << endl;
	}cout << endl; cout << endl;
	}


	*/


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

	const int outputDimHW = 8;

	//Convolution행렬 선언
	//float Output_Conv[ImageNum][FeatureNum][outputDimHW][outputDimHW];


	//GPU에 Convolution 결과 행렬 할당
	float* dev_Output_Conv;
	cudaMalloc((void**)&dev_Output_Conv, sizeof(float) * out_conv_c * out_conv_h * out_conv_n * out_conv_w);
	//Convolution 구조체 초기화
	cudnnSetTensor4dDescriptor(out_conv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, ImageNum, FeatureNum, outputDim, outputDim);
	//입력과 필터, 컨볼루션 패딩, 스트라이드가 위와 같이 주어졌을때 가장 빠른 알고리즘이 무엇인지를 알아내기
	cudnnConvolutionFwdAlgo_t alg;
	alg = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
	//Conv 버퍼 데이터크기 알아내는 법 및 버퍼 메모리 할당 추가
	size_t WS_size = 0;
	cudnnGetConvolutionForwardWorkspaceSize(// This function returns the amount of GPU memory workspace
		cudnn, in_desc, filt_desc, conv_desc, out_conv_desc, alg, &WS_size);
	size_t* dev_WS;
	cudaMalloc((void**)&dev_WS, WS_size);
	//연산
	float alpha = 1.0;
	float beta = 0.0;

	float Output_Conv[ImageNum][FeatureNum][outputDimHW][outputDimHW];

	//Convolution결과 GPU로 복사


	//********
	//**Bias**
	//********
	beta = 1.0f;
	//Bias 결과 저장행렬 선언
	//float Output_Bias[ImageNum][FeatureNum][outputDimHW][outputDimHW];
	//bias 값 지정
	//float biasValue[filt_n] = { -10.0f };


	std::vector<float> biasValue;
	biasValue.resize(filt_n);
	InitWeightsbias(filt_n, biasValue);

	//GPU에 bias값 복사
	float* dev_Bias;
	cudaMalloc((void**)&dev_Bias, sizeof(float));
	cudaMemcpy(dev_Bias, &biasValue[0], sizeof(float), cudaMemcpyHostToDevice);
	//bias결과 저장행렬 선언, 할당
	cudnnTensorDescriptor_t bias_desc;
	cudnnCreateTensorDescriptor(&bias_desc);
	cudnnSetTensor4dDescriptor(bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, filt_n, 1, 1);
	//bias 덧셈 수행

	//Bias합 결과
	//cudaMemcpy(Output_Bias, dev_Output_Conv, sizeof(float) * ImageNum * FeatureNum * outputDimHW * outputDimHW, cudaMemcpyDeviceToHost);

	//***********************
	//**Actiovation Funtion**
	//***********************
	beta = 0.0;
	//Activation Function 구조체 선언 및 할당
	cudnnActivationDescriptor_t act_desc;
	cudnnCreateActivationDescriptor(&act_desc);
	//Activation Function 종류 지정 - 추후 조정가능하도록
	cudnnActivationMode_t Activation_Function;
	//Activation_Function = CUDNN_ACTIVATION_RELU;
	//Activation_Function = CUDNN_ACTIVATION_TANH;
	Activation_Function = CUDNN_ACTIVATION_SIGMOID;
	cudnnSetActivationDescriptor(act_desc, Activation_Function, CUDNN_PROPAGATE_NAN, 0);
	//Activation Function 메모리 GPU에 복사
	float* dev_Output_Act;
	cudaMalloc((void**)&dev_Output_Act, sizeof(float) * outputDimHW * outputDimHW * 3 * 3);
	//Activatin Function 연산수행


	//Activation Function 결과값 저장 행렬
	//float Output_Activation[ImageNum][FeatureNum][outputDimHW][outputDimHW];
	//cudaMemcpy(Output_Activation, dev_Output_Act, sizeof(float) * ImageNum * FeatureNum * outputDimHW * outputDimHW, cudaMemcpyDeviceToHost);
	//Actavation Function 결과

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
	cudnnGetPooling2dForwardOutputDim(pool_desc, out_conv_desc, &out_pool_n, &out_pool_c, &out_pool_h, &out_pool_w);
	//GPU에 Pooling 결과행렬 메모리할당
	float* dev_Output_Pool;
	cudaMalloc((void**)&dev_Output_Pool, sizeof(float) * out_pool_n * out_pool_c * out_pool_h * out_pool_w);
	//Pooling 저장행렬 구조체 초기화
	cudnnSetTensor4dDescriptor(out_pool_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, out_pool_n, out_pool_c, out_pool_h, out_pool_w);
	//Pooling연산 수행

	//Pooling결과
	//float Output_Pool[ImageNum][FeatureNum][(outputDimHW + 2 * pool_pad_h) / pool_strd_h][(outputDimHW + 2 * pool_pad_w) / pool_strd_w];
	//cudaMemcpy(Output_Pool, dev_Output_Pool, sizeof(float) * out_pool_n * out_pool_c * out_pool_h * out_pool_w, cudaMemcpyDeviceToHost);

	//Pooling행렬
	//*******************
	//**Fully Connected**
	//*******************
	/*
	//Weights 선언
	float Weights[10][3][4][4];

	//Weights 정의
	for (int och = 0; och < 10; och++)
	{
	for (int ch = 0; ch < 3; ch++)
	{
	for (int row = 0; row < 4; row++)
	{
	for (int col = 0; col < 4; col++)
	{
	Weights[och][ch][row][col] = (float)(row + col + och + ch) * 0.11;
	}
	}
	}
	}
	*/

	//std::vector<float> Weights; //[lastLayer_numOut][filterCnt][poolOutHeight][poolOutWidth];
	//Weights.resize(out_pool_n * out_pool_c * out_pool_h * out_pool_w); ////in_channels_ * kernel_size_ * kernel_size_ * out_channels_
	//InitWeightsXavier(out_pool_n, out_pool_c, out_pool_n * out_pool_c * out_pool_h * out_pool_w, Weights);

	/*

	for (int ch = 0; ch < 3 * 10 * 4 * 4; ch++)
	{
	if (ch % 4 == 0)
	cout << endl;
	cout << setw(10) << Weights[ch] << "::";

	}
	*/

	//GPU에 Weights행렬 복사
	float* dev_weights;
	cudaMalloc((void**)&dev_weights, sizeof(float) * 10 * 3 * 4 * 4);
	cudaMemcpy(dev_weights, Weights, sizeof(float) * 10 * 3 * 4 * 4, cudaMemcpyHostToDevice);
	//Weights를 위한 Filter 구조체 선언 및 할당
	cudnnFilterDescriptor_t weights_desc;
	cudnnCreateFilterDescriptor(&weights_desc);
	cudnnSetFilter4dDescriptor(weights_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 10, 3, 4, 4);
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
	cudnnGetConvolution2dForwardOutputDim(fc_desc, out_fc_desc, weights_desc, &out_fc_n, &out_fc_c, &out_fc_h, &out_fc_w);
	//FC 결과행렬 선언

	//GPU에 FC 결과행렬 할당
	float* dev_Output_FC;
	cudaMalloc((void**)&dev_Output_FC, sizeof(float) * ImageNum * 10 * 1 * 1);
	//FC 구조체 초기화
	cudnnSetTensor4dDescriptor(out_fc_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, ImageNum, 10, 1, 1);
	//FC 버퍼크기 할당 및 도출
	size_t WS_size2 = 0;
	cudnnGetConvolutionForwardWorkspaceSize(cudnn, out_pool_desc, weights_desc, fc_desc, out_fc_desc, alg, &WS_size2);
	size_t* dev_WS2;
	cudaMalloc((void**)&dev_WS2, WS_size2);
	//Fully Connected 연산

	//FC 결과를 CPU에 저장

	//*************************
	//**Fully Conncected Bias**
	//*************************

	const int ten = 10;
	/*
	//FC bias값
	float biasValueFC[10];

	for (int i = 0; i < ten; i++)
	{
	biasValueFC[i] = -5.0f;
	}
	*/

	std::vector<float> biasValueFC;
	biasValueFC.resize(10);
	InitWeightsbias(10, biasValueFC);

	//GPU에 FC bias값 복사
	float* dev_Bias_FC;
	cudaMalloc((void**)&dev_Bias_FC, sizeof(float) * 10);
	cudaMemcpy(dev_Bias_FC, &biasValueFC[0], sizeof(float) * 10, cudaMemcpyHostToDevice);

	cudnnTensorDescriptor_t out_Bias_FC_desc;
	cudnnCreateTensorDescriptor(&out_Bias_FC_desc);
	cudnnSetTensor4dDescriptor(out_Bias_FC_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, ten, 1, 1);
	//bias 덧셈 수행

	//***********
	//**Softmax**
	//***********

	float* dev_Output_Softmax;
	cudaMalloc((void**)&dev_Output_Softmax, sizeof(float) * ImageNum * 10);

	//****************************
	//****************************
	//**Backpropagation 연산수행**
	//****************************

	//***************************
	//**Softmax Backpropagation** - p - y
	//***************************

	//저장행렬 선언
	//float SoftBack[ImageNum][10][1][1];
	//GPU 메모리 할당
	float* dif_Soft_Back;
	cudaMalloc((void**)&dif_Soft_Back, sizeof(float) * ImageNum * 10);
	//구조체 선언 및 초기화
	cudnnTensorDescriptor_t dif_soft_desc;
	cudnnCreateTensorDescriptor(&dif_soft_desc);
	cudnnSetTensor4dDescriptor(dif_soft_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, ImageNum, 10, 1, 1);

	//delta
	float* dev_Output_Soft_Back;
	cudaMalloc((void**)&dev_Output_Soft_Back, sizeof(float) * ImageNum * 10);
	//cudaMemcpy(dev_Output_Soft_Back, dy, sizeof(float) * ImageNum * 10, cudaMemcpyHostToDevice);

	cudnnTensorDescriptor_t dif_soft_back;
	cudnnCreateTensorDescriptor(&dif_soft_back);
	cudnnSetTensor4dDescriptor(dif_soft_back, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, ImageNum, 10, 1, 1);


	//*********************************
	//**Fully Connected Bias Backward**
	//*********************************   Output_FC_Bias
	//GPU 메모리
	float* dev_FC_bias_Back;
	cudaMalloc((void**)&dev_FC_bias_Back, sizeof(float));

	//**********************************
	//**Fc의 W delta Backpropagtion**
	//**********************************

	float* dev_Filter_Gradient;
	cudaMalloc((void**)&dev_Filter_Gradient, sizeof(float) * 3 * 10 * 4 * 4);
	// Workspace
	size_t WS_size3 = 0;
	cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn, weights_desc, dif_soft_back, fc_desc, out_pool_desc, CUDNN_CONVOLUTION_BWD_DATA_ALGO_0, &WS_size3);
	//GPU에 workspace 메모리 할당
	size_t* dev_WS3;
	cudaMalloc((void**)&dev_WS3, WS_size3);

	//**********************************
	//**fc 바로전, pooling 결과값 delta Backpropagtion**
	//**********************************

	float* dev_data_Gradient;
	cudaMalloc((void**)&dev_data_Gradient, sizeof(float) * ImageNum * 3 * 4 * 4);


	//***************************
	//**Pooling Backpropagation**
	//***************************

	float* dev_Pool_Back;
	cudaMalloc((void**)&dev_Pool_Back, sizeof(float) *ImageNum * 3 * 8 * 8);
	//구조체 선언부터 초기화
	cudnnTensorDescriptor_t pool_back_desc;
	cudnnCreateTensorDescriptor(&pool_back_desc);
	cudnnSetTensor4dDescriptor(pool_back_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, ImageNum, 3, 8, 8);

	//******************************
	//**Activation Backpropagation**
	//******************************
	float* dev_Act_Back;
	cudaMalloc((void**)&dev_Act_Back, sizeof(float) *ImageNum * 3 * 8 * 8);


	//*********************************
	//**Conv Bias Backward**
	//*********************************   Output_Conv_Bias

	//GPU 메모리
	float* dev_Conv_bias_Back;
	cudaMalloc((void**)&dev_Conv_bias_Back, sizeof(float) * ImageNum * 3);


	//******************************
	//**Conv filter Backpropagation**
	//******************************

	float* dev_Filter_conv2;
	cudaMalloc((void**)&dev_Filter_conv2, sizeof(float) * 3 * 3 * 8 * 8);
	// Workspace
	size_t WS_size4 = 0;
	cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn, filt_desc, out_conv_desc, fc_desc, in_desc, CUDNN_CONVOLUTION_BWD_DATA_ALGO_0, &WS_size4);
	//GPU에 workspace 메모리 할당
	size_t* dev_WS4;
	cudaMalloc((void**)&dev_WS4, WS_size4);


	float OutSoft[ImageNum][10][1][1];



	for (int iter = 0; iter < 10; ++iter)
	{

		//컨볼류
		cudnnConvolutionForward(cudnn, &alpha, in_desc, dev_Input, filt_desc, dev_Filt, conv_desc, alg, dev_WS, WS_size, &beta, out_conv_desc, dev_Output_Conv);


		cudaMemcpy(Output_Conv, dev_Output_Conv, sizeof(float) * out_conv_n* out_conv_c* out_conv_h* out_conv_w, cudaMemcpyDeviceToHost);

		std::cout << std::endl << std::endl << "Convolution 결과" << std::endl << std::endl;
		for (int i = 0; i < out_conv_h; i++)
		{
			for (int j = 0; j < out_conv_w; j++)
			{
				std::cout << setw(10) << Output_Conv[0][0][i][j] << "  ";
			}
			std::cout << std::endl;
		}


		//bias 계산


		beta = 1.0f;
		cudnnAddTensor(cudnn, &alpha, bias_desc, dev_Bias, &beta, out_conv_desc, dev_Output_Conv);

		float Output_Bias[ImageNum][FeatureNum][outputDimHW][outputDimHW];

		cudaMemcpy(Output_Bias, dev_Output_Conv, sizeof(float) * ImageNum * FeatureNum * outputDimHW * outputDimHW, cudaMemcpyDeviceToHost);

		std::cout << std::endl << std::endl << "Add Bias (bias : 1)" << std::endl << std::endl;
		for (int i = 0; i < out_conv_h; i++)
		{
			for (int j = 0; j < out_conv_w; j++)
			{
				std::cout << setw(10) << Output_Bias[0][0][i][j] << "  ";
			}
			std::cout << std::endl;
		}


		// activation 

		beta = 0;
		cudnnActivationForward(cudnn, act_desc, &alpha, out_conv_desc, dev_Output_Conv, &beta, out_conv_desc, dev_Output_Act);

		float Output_Activation[ImageNum][FeatureNum][outputDimHW][outputDimHW];

		cudaMemcpy(Output_Activation, dev_Output_Act, sizeof(float) * ImageNum * FeatureNum * outputDimHW * outputDimHW, cudaMemcpyDeviceToHost);


		std::cout << std::endl << std::endl << "Activation Function 결과" << std::endl << std::endl;

		for (int i = 0; i < outputDimHW; i++)
		{
			for (int j = 0; j < outputDimHW; j++)
			{
				std::cout << setw(5) << Output_Activation[0][0][i][j] << "  ";
			}
			std::cout << std::endl;
		}


		// pooling 계산
		cudnnPoolingForward(cudnn, pool_desc, &alpha, out_conv_desc, dev_Output_Act, &beta, out_pool_desc, dev_Output_Pool);
		float Output_Pool[ImageNum][FeatureNum][(outputDimHW + 2 * pool_pad_h) / pool_strd_h][(outputDimHW + 2 * pool_pad_w) / pool_strd_w];
		cudaMemcpy(Output_Pool, dev_Output_Pool,
			sizeof(float) * out_pool_n * out_pool_c * out_pool_h * out_pool_w, cudaMemcpyDeviceToHost);

		std::cout << std::endl << std::endl << "Pooling 결과" << std::endl << std::endl;

		for (int i = 0; i < out_pool_h; i++)
		{
			for (int j = 0; j < out_pool_w; j++)
			{
				std::cout << setw(5) << Output_Pool[0][0][i][j] << "  ";
			}
			std::cout << std::endl;
		}


		//Fully Connected 연산
		cudnnConvolutionForward(cudnn, &alpha, out_pool_desc, dev_Output_Pool, weights_desc, dev_weights, fc_desc, alg, dev_WS2, WS_size2, &beta, out_fc_desc, dev_Output_FC);

		float Output_FC[ImageNum][10][1][1];
		cudaMemcpy(Output_FC, dev_Output_FC, sizeof(float) * ImageNum * 10 * 1 * 1, cudaMemcpyDeviceToHost);

		std::cout << std::endl << std::endl << "Fully Connected  결과" << std::endl << std::endl;

		for (int n = 0; n < ImageNum; n++)
		{
			for (int c = 0; c < 10; c++) {

				cout << setw(10) << Output_FC[n][c][0][0] << "::";

			}cout << endl;

		}

		//FC bias 덧셈 수행
		beta = 1.0f;
		cudnnAddTensor(cudnn, &alpha, out_Bias_FC_desc, dev_Bias_FC, &beta, out_fc_desc, dev_Output_FC);
		float Output_FC_Bias[ImageNum][10][1][1];
		cudaMemcpy(Output_FC_Bias, dev_Output_FC, sizeof(float) * ImageNum * ten, cudaMemcpyDeviceToHost);

		std::cout << std::endl << std::endl << "FC bias 결과" << std::endl << std::endl;

		for (int n = 0; n < ImageNum; n++)
		{
			for (int c = 0; c < 10; c++) {

				cout << setw(10) << Output_FC_Bias[n][c][0][0] << "::";

			}cout << endl;

		}


		beta = 0;
		//softmax 


		cudnnSoftmaxForward(cudnn, CUDNN_SOFTMAX_FAST, CUDNN_SOFTMAX_MODE_INSTANCE, &alpha, out_fc_desc, dev_Output_FC, &beta, out_fc_desc, dev_Output_Softmax);

		std::cout << std::endl << std::endl << "softmax 결과" << std::endl << std::endl;

		cudaMemcpy(OutSoft, dev_Output_Softmax, sizeof(float) * ImageNum * 10, cudaMemcpyDeviceToHost);
		for (int n = 0; n < ImageNum; n++)
		{
			for (int c = 0; c < 10; c++) {

				cout << setw(10) << OutSoft[n][c][0][0] << "::";

			}cout << endl;

		}

		for (int n = 0; n < ImageNum; n++)
		{
			for (int c = 0; c < 10; c++) {
				cout << setw(10) << TargetY[n][c] << "::";

			}cout << endl;

		}

		for (int n = 0; n < ImageNum; n++)
		{

			cout << setw(10) << (labels[n] + 1) << "::";



		}cout << endl;


		// cost function (크로스 엔트로피)으로 오차 계산 
		float sum2 = 0;
		float sum3 = 0;
		for (int n = 0; n < ImageNum; n++)
		{
			for (int c = 0; c < 10; c++) {

				sum3 = sum3 + (OutSoft[n][c][0][0] * TargetY[n][c]);

				sum2 = sum2 + (-log(OutSoft[n][c][0][0]) * TargetY[n][c]);
			}cout << endl;

			cout << "error(크로스엔트로피 cost) :: " << sum2 << endl;
			cout << "정확도 :: " << sum3 << "  %" << endl;
		}






		float suma[ImageNum];
		float sum = 0;
		for (int n = 0; n < ImageNum; n++)
		{
			for (int c = 0; c < 10; c++) {


				sum = sum + OutSoft[n][c][0][0];

			}
			cout << endl;

			suma[n] = sum;

			cout << suma[n] << "< = 1이 였으면 좋겠다" << endl;
		}

		// 에러 처리 
		/*

		float  *loss;
		cudaMalloc((void**)&loss, sizeof(float) * numImgs * 10 * 1 * 1);
		// 라벨, 라벨수, 이미지 총수, 소프트맥스 결과
		SoftmaxLossBackprop <<< RoundUp(100, BW), BW >>> (d_labels, ten, ImageNum, dev_Output_Softmax, loss);
		float Output_loss[ImageNum][10][1][1];
		cudaMemcpy(Output_loss, loss, sizeof(float) * ImageNum * ten, cudaMemcpyDeviceToHost);
		*/



		float Output_loss[ImageNum][10];
		for (int n = 0; n < ImageNum; n++)
		{
			for (int c = 0; c < 10; c++) {
				Output_loss[n][c] = (TargetY[n][c] - OutSoft[n][c][0][0]);
			}
		}

		float * loss;
		cudaMalloc((void**)&loss, sizeof(float) * ImageNum * 10);
		cudaMemcpy(loss, Output_loss, sizeof(float) * ImageNum * 10, cudaMemcpyHostToDevice);



		cout << "오차차차차차차 Output_dloss_data" << endl;
		for (int n = 0; n < ImageNum; n++)
		{
			for (int c = 0; c < 10; c++) {

				cout << setw(10) << Output_loss[n][c] << "::";

			}cout << endl;

		}



		beta = 1.0f;
		//FC BACK bias delta 계산
		cudnnConvolutionBackwardBias(cudnn, &alpha, out_fc_desc, loss, &beta, bias_desc, dev_FC_bias_Back);
		beta = 0;

		//FC Weight delta 계산
		cudnnConvolutionBackwardFilter(cudnn, &alpha, out_pool_desc, dev_Output_Pool, dif_soft_back, loss, fc_desc, CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0, dev_WS3, WS_size3, &beta, weights_desc, dev_Filter_Gradient);

		//FC X Delta 계산  
		cudnnConvolutionBackwardData(cudnn, &alpha, weights_desc, dev_weights, dif_soft_back, loss, fc_desc, CUDNN_CONVOLUTION_BWD_DATA_ALGO_0, dev_WS3, WS_size3, &beta, out_pool_desc, dev_data_Gradient);

		//Pooling delta계산
		cudnnPoolingBackward(cudnn, pool_desc, &alpha, out_pool_desc, dev_Output_Pool, out_pool_desc, dev_data_Gradient, out_conv_desc, dev_Output_Act, &beta, pool_back_desc, dev_Pool_Back);

		//Activat delta계산
		cudnnActivationBackward(cudnn, act_desc, &alpha, out_conv_desc, dev_Output_Act, out_conv_desc, dev_Pool_Back, out_conv_desc, dev_Output_Conv, &beta, out_conv_desc, dev_Act_Back);

		beta = 1.0f;
		//Bias delta 계산
		cudnnConvolutionBackwardBias(cudnn, &alpha, out_conv_desc, dev_Act_Back, &beta, bias_desc, dev_Conv_bias_Back);
		beta = 0;
		//Weight delta 계산
		cudnnConvolutionBackwardFilter(cudnn, &alpha, in_desc, dev_Input, out_conv_desc, dev_Act_Back, conv_desc, CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0, dev_WS4, WS_size4, &beta, filt_desc, dev_Filter_conv2);


		//weight update
		float learning_rate = -0.1f;
		// Conv1
		cublasSaxpy(cublasHandle, static_cast<int>(filt_n * filt_c * filt_h * filt_w),
			&learning_rate, dev_Filter_conv2, 1, dev_Filt, 1);
		cublasSaxpy(cublasHandle, static_cast<int>(1 * filt_n * 1 * 1),
			&learning_rate, dev_Conv_bias_Back, 1, dev_Bias, 1);

		// Fully connected 1
		cublasSaxpy(cublasHandle, static_cast<int>(10 * 3 * 4 * 4),
			&learning_rate, dev_Filter_Gradient, 1, dev_weights, 1);
		cublasSaxpy(cublasHandle, static_cast<int>(1 * 10 * 1 * 1),
			&learning_rate, dev_FC_bias_Back, 1, dev_Bias_FC, 1);

		std::cout << iter << endl;

		float *Filter2;
		cudaMalloc(&Filter2, sizeof(float) * 3 * 3 * 8 * 8);



		cudaMemcpy(Filter, dev_Filt, sizeof(float) * 3 * 3 * 8 * 8, cudaMemcpyDeviceToHost);


		std::cout << std::endl << std::endl << "Filter_conv[] " << std::endl << std::endl;
		//std::cout << setw(8) << Filter_conv[0][0][0][0] << " :: ";
		std::cout << std::endl;
		for (int n = 0; n < 1; n++)
		{
			for (int c = 0; c < 3; c++)
			{
				for (int i = 0; i < 8; i++)
				{
					for (int j = 0; j < 8; j++)
					{
						std::cout << setw(10) << Filter[n][c][i][j] << " :: ";
					}
					std::cout << std::endl;
				}
				std::cout << std::endl;
			}
			std::cout << "==========================================" << endl;
			std::cout << std::endl; std::cout << std::endl;
		}

		cudaMemcpy(Weights, dev_weights, sizeof(float) * 10 * 3 * 4 * 4, cudaMemcpyDeviceToHost);


		std::cout << std::endl << std::endl << "WEIGHT[] " << std::endl << std::endl;

		std::cout << std::endl;
		for (int n = 0; n < 1; n++)
		{
			for (int c = 0; c < 3; c++)
			{
				for (int i = 0; i < 4; i++)
				{
					for (int j = 0; j < 4; j++)
					{
						std::cout << setw(10) << Weights[n][c][i][j] << " :: ";
					}
					std::cout << std::endl;
				}
				std::cout << std::endl;
			}
			std::cout << "==========================================" << endl;
			std::cout << std::endl; std::cout << std::endl;
		}
		/*
		std::cout << std::endl << std::endl << "Filter2 vect" << std::endl << std::endl;
		for (int ch = 0; ch < 3 * 3 * 8 * 8; ch++)
		{
		if (ch % 8 == 0)
		cout << endl;
		cout << setw(10) << Filter2[ch] << "::";

		}
		*/

		/*
		//int num_errors = 0;
		int chosen = 0;
		for (int id = 1; id < 10; ++id)
		{
		if (OutSoft[0][chosen][0][0] < OutSoft[0][id][0][0]) chosen = id;
		}
		if (chosen != labels[0])
		cout << "no 못 맞춤 ㅠㅠ" << endl;
		//++num_errors;



		// cost function (크로스 엔트로피)으로 오차 계산
		float sum2 = 0;

		for (int c = 0; c < 10; c++) {

		sum2 = sum2 + (-log(OutSoft[0][c][0][0]) * (float)TargetY[0][c]);

		}

		cout << "error :: " << sum2 << endl;

		sum2 = 0;
		*/
		//int num_errors = 0;
		for (int n = 0; n < ImageNum; n++)
		{
			int chosen = 0;
			for (int id = 1; id < 10; ++id)
			{
				if (OutSoft[n][chosen][0][0] < OutSoft[n][id][0][0]) chosen = id;
			}
			if (chosen != labels[0])
				cout << "no 못 맞춤 ㅠㅠ" << endl;
			//++num_errors;

		}

	}



	std::cout << "끝==========================================" << endl;

	cudaFree(dev_Filter_conv2);
	cudaFree(dev_Conv_bias_Back);
	cudaFree(dev_Act_Back);
	cudaFree(dev_Pool_Back);
	cudaFree(dev_data_Gradient);
	cudaFree(dev_WS3);
	cudaFree(dev_Filter_Gradient);
	cudaFree(dev_FC_bias_Back);
	cudaFree(dev_Output_Soft_Back);
	cudaFree(dif_Soft_Back);
	cudaFree(dev_Output_Softmax);
	cudaFree(dev_Bias_FC);
	cudaFree(dev_WS2);
	cudaFree(dev_Output_FC);
	cudaFree(dev_weights);
	cudaFree(dev_Output_Pool);
	cudaFree(dev_Output_Act);
	cudaFree(dev_Bias);
	cudaFree(dev_WS);
	cudaFree(dev_Output_Conv);
	cudaFree(dev_Filt);
	cudaFree(dev_Input);


}