#include <math.h>
#include <iostream>
#include <iomanip>
#include <cuda.h>
#include <cudnn.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cassert>
#include <io.h>
#include <vector>
#include <random>
#include <cublas_v2.h>

#include <stdio.h>
#include <string.h>
#include <opencv2/opencv.hpp>
#include <device_launch_parameters.h>

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



__global__ void SoftmaxLossBackprop(const float *label, int num_labels, int batch_size, float *diff)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= batch_size)
		return;

	const int label_value = static_cast<int>(label[idx]);

	// For each item in the batch, decrease the result of the label's value by 1
	diff[idx * num_labels + label_value] -= 1.0f;
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

void checkCUDNN(cudnnStatus_t status)
{
	if (status != CUDNN_STATUS_SUCCESS)
		std::cout << "[ERROR] CUDNN " << status << std::endl;
}

void checkCudaErrors(cudaError_t error)
{
	if (error != CUDA_SUCCESS)
		std::cout << "[ERROR] CUDA " << error << std::endl;
}

void InitWeightsXavierUni(const size_t & PrevLayerNodeNumber, const size_t & NextLayerNodeNumber, const size_t TotalNumber, std::vector<float> & Weights)
{
	/*
	const size_t BiasNodeNumber = 1;

	float min = -sqrt(6 / (PrevLayerNodeNumber + NextLayerNodeNumber + BiasNodeNumber));
	float max = sqrt(6 / (PrevLayerNodeNumber + NextLayerNodeNumber + BiasNodeNumber));

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> dist(min, max);

	for (size_t iter = 0; iter < TotalNumber; iter++)
	{
	Weights[iter] = static_cast<float>(dist(gen));
	}
	*/

	float sigma = sqrt(2 / (float)(PrevLayerNodeNumber + NextLayerNodeNumber));

	random_device rd;
	mt19937 gen(rd());
	normal_distribution<float> d(0, sigma);

	for (size_t iter = 0; iter < TotalNumber; iter++)
	{

		Weights[iter] = d(gen);
	}

}

void InitWeightsbias(const size_t & numOutSize, std::vector<float> & Weightsbias)
{
	for (int i = 0; i < numOutSize; i++)
	{
		Weightsbias[i] = 0.0f;
	}
}


int main()
{
	const int batchSize = 1; // 이미지 총 갯수
	string folder_path = "C:\\Users\\ECMUser\\Desktop\\cifarTest"; // 이미지가 저장되어 있는 폴더 경로
	vector<pair<Mat, string>> ImgBox; // 이미지 데이터, 이미지 이름
	ImgBox = TraverseFilesUsingDFS(folder_path);
	vector<string> LabelBox; // 라벨 정리를 위해 생성
	vector<pair<int, string>> LabelTable; // 라벨링 마다 넘버 부여
	vector<pair<Mat, int>> ImgBox2; // 이미지 데이터, 라벨 넘버
	vector<float> labels(batchSize);
	vector<vector<int>> TargetY; // 라벨 넘버 -> 이진수 형태 데이터로 저장

								 // 라벨에 번호 부여를 위해 LabelBox 벡터에 값 복사 하고 정렬 및 중복 삭제
	for (int i = 0; i < batchSize; i++)
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
	for (int i = 0; i < batchSize; i++)
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

	float  *d_labels;
	cudaMalloc((void**)&d_labels, sizeof(float) * batchSize * 10 * 1 * 1);

	cudaMemcpyAsync(d_labels, &labels, sizeof(float) * batchSize, cudaMemcpyHostToDevice);



	// TargetY 셋팅, 정답 이진수 형태로 표현
	TargetY.resize(batchSize);

	for (int i = 0; i < batchSize; i++)
	{
		TargetY[i].resize(nLabelBoxSize, 0);
	}

	for (int i = 0; i < batchSize; i++)
	{
		int idx = ImgBox2[i].second;
		TargetY[i][idx] = 1;
	}

	/*
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

	// mat 형식 - > 4차 행렬
	for (int i = 0; i < batchSize; i++)
	{
	unsigned char* temp = ImgBox2[i].first.data;

	for (int c = 0; c < 3; c++)
	{
	for (int y = 0; y < 32; y++)
	{
	for (int x = 0; x < 32; x++)
	{
	Input[i][c][y][x] = temp[3 * 32 * y + 3 * x + c];
	}
	}
	}
	}
	*/
	vector<float> Input;
	Input.resize(batchSize * 3 * 32 * 32);

	for (int i = 0; i < batchSize; i++)
	{
		auto pDATA = ImgBox2[i].first.data;
		for (int j = 0; j < batchSize * 3 * 32 * 32; j++)
		{
			Input[j] = ((2 * (float)pDATA[j]) / 255) - 1;
			//Input[j] = pDATA[j];
		}
	}


	for (int j = 0; j < batchSize * 3 * 32 * 32; j++)
	{
		cout << Input[j] << endl;

	}




	cudnnHandle_t cudnn;
	checkCUDNN(cudnnCreate(&cudnn));
	cublasHandle_t cublasHandle;
	cublasCreate(&cublasHandle);

	float alpha = 1.0;
	float beta = 0.0;

	const int input_channelCnt = 3, imageHeight = 32, imageWidth = 32;
	const int filterCnt = 5, convFilterHeight = 3, convFilterWidth = 3;
	const int convPad_h = 1, convPad_w = 1, convStr_h = 1, convStr_w = 1, convDil_h = 1, convDil_w = 1;
	const int poolWind_h = 2, poolWind_w = 2, poolPad_h = 0, poolPad_w = 0, poolStrd_w = 2, poolStrd_h = 2;
	const int fcPad_h = 0, fcPad_w = 0, fcStr_h = 1, fcStr_w = 1, fcDil_h = 1, fcDil_w = 1;
	const int convOutHeight = ((imageHeight + (2 * convPad_h) - (((convFilterHeight - 1) * convDil_h) + 1)) / convStr_h) + 1;
	const int convOutWidth = ((imageWidth + (2 * convPad_w) - (((convFilterWidth - 1) * convDil_w) + 1)) / convStr_w) + 1;
	const int poolOutHeight = ((convOutHeight + (2 * poolPad_h) - poolWind_h) / poolStrd_h) + 1;
	const int poolOutWidth = ((convOutWidth + (2 * poolPad_w) - poolWind_w) / poolStrd_w) + 1;
	const int lastLayer_numOut = 10; // 출력 클래스 수


	float m_Outputdata[batchSize][filterCnt][imageHeight][imageWidth];


	std::vector<float> convFilter; //[filterCnt][input_channelCnt][convFilterHeight][convFilterWidth]
	convFilter.resize(input_channelCnt * convFilterHeight * convFilterWidth * filterCnt); //in_channels_ * kernel_size_ * kernel_size_ * out_channels_
	InitWeightsXavierUni(input_channelCnt, filterCnt, input_channelCnt * convFilterHeight * convFilterWidth * filterCnt, convFilter);

	for (int ch = 0; ch < 5 * 3 * 3 * 3; ch++)
	{
		if (ch % 8 == 0)
			cout << endl;
		cout << setw(10) << convFilter[ch] << "::";

	}
	cout << endl << "111" << endl;


	std::vector<float> fcFilter; //[lastLayer_numOut][filterCnt][poolOutHeight][poolOutWidth];
	fcFilter.resize(filterCnt * poolOutHeight * poolOutWidth * lastLayer_numOut); ////in_channels_ * kernel_size_ * kernel_size_ * out_channels_
	InitWeightsXavierUni(filterCnt, lastLayer_numOut, filterCnt * poolOutHeight * poolOutWidth * lastLayer_numOut, fcFilter);

	for (int ch = 0; ch < filterCnt * poolOutHeight * poolOutWidth * lastLayer_numOut; ch++)
	{
		if (ch % 16 == 0)
			cout << endl;
		cout << setw(10) << fcFilter[ch] << "::";

	}
	cout << endl << "fcfilterer" << endl;



	std::vector<float> convBias;
	convBias.resize(filterCnt);
	InitWeightsbias(filterCnt, convBias);

	std::vector<float> fcBias;
	fcBias.resize(lastLayer_numOut);
	InitWeightsbias(lastLayer_numOut, fcBias);


	float* dev_input;
	checkCudaErrors(cudaMalloc((void**)&dev_input, sizeof(float) * batchSize * input_channelCnt * imageHeight * imageWidth));
	checkCudaErrors(cudaMemcpy(dev_input, &Input[0], sizeof(float) * batchSize * input_channelCnt * imageHeight * imageWidth, cudaMemcpyHostToDevice));




	float* dev_convFilter;
	checkCudaErrors(cudaMalloc((void**)&dev_convFilter, sizeof(float) * filterCnt * input_channelCnt * convFilterHeight * convFilterWidth));
	checkCudaErrors(cudaMemcpy(dev_convFilter, &convFilter[0], sizeof(float) * filterCnt * input_channelCnt * convFilterHeight * convFilterWidth, cudaMemcpyHostToDevice));

	std::vector<float> Filter3;
	Filter3.resize(input_channelCnt * convFilterHeight * convFilterWidth * filterCnt);
	cudaMemcpy(&Filter3[0], dev_convFilter, sizeof(float) * filterCnt * input_channelCnt * convFilterHeight * convFilterWidth, cudaMemcpyDeviceToHost);

	for (int ch = 0; ch < 5 * 3 * 3 * 3; ch++)
	{
		if (ch % 8 == 0)
			cout << endl;
		cout << setw(10) << Filter3[ch] << "::";

	}
	cout << endl << "222" << endl;

	float* dev_convOutput;
	checkCudaErrors(cudaMalloc((void**)&dev_convOutput, sizeof(float) * batchSize * filterCnt * convOutHeight * convOutWidth));
	float* dev_convBias;
	checkCudaErrors(cudaMalloc((void**)&dev_convBias, sizeof(float) * 1 * filterCnt * 1 * 1));
	checkCudaErrors(cudaMemcpy(dev_convBias, &convBias[0], sizeof(float) * 1 * filterCnt * 1 * 1, cudaMemcpyHostToDevice));

	float* dev_output_Act;
	checkCudaErrors(cudaMalloc((void**)&dev_output_Act, sizeof(float) * batchSize * filterCnt * convOutHeight * convOutWidth));


	float* dev_poolOutput;
	checkCudaErrors(cudaMalloc((void**)&dev_poolOutput, sizeof(float) * batchSize * filterCnt * poolOutHeight * poolOutWidth));


	float* dev_fcFilter;
	checkCudaErrors(cudaMalloc((void**)&dev_fcFilter, sizeof(float) * lastLayer_numOut * filterCnt * poolOutHeight * poolOutWidth));
	checkCudaErrors(cudaMemcpy(dev_fcFilter, &fcFilter[0], sizeof(fcFilter), cudaMemcpyHostToDevice));


	float* dev_fcOutput;
	checkCudaErrors(cudaMalloc((void**)&dev_fcOutput, sizeof(float) * batchSize * lastLayer_numOut * 1 * 1));

	float* dev_fcBias;
	checkCudaErrors(cudaMalloc((void**)&dev_fcBias, sizeof(float) * 1 * lastLayer_numOut * 1 * 1));
	checkCudaErrors(cudaMemcpy(dev_fcBias, &fcBias[0], sizeof(float) * 1 * lastLayer_numOut * 1 * 1, cudaMemcpyHostToDevice));

	float* dev_smaxOutput;
	checkCudaErrors(cudaMalloc((void**)&dev_smaxOutput, sizeof(float) * batchSize * lastLayer_numOut * 1 * 1));


	cudnnTensorDescriptor_t input_Tensor, conv_Tensor, bias_convTensor, poolOutTensor, fcTensor, bias_fcTensor;
	cudnnFilterDescriptor_t convFilter_Desc, fcFilter_Desc;
	cudnnConvolutionDescriptor_t conv_Desc, fc_Desc;
	cudnnActivationDescriptor_t convAct_Desc;
	cudnnPoolingDescriptor_t pool_Desc;


	checkCUDNN(cudnnCreateTensorDescriptor(&input_Tensor));
	checkCUDNN(cudnnCreateTensorDescriptor(&conv_Tensor));
	checkCUDNN(cudnnCreateTensorDescriptor(&bias_convTensor));
	checkCUDNN(cudnnCreateTensorDescriptor(&poolOutTensor));
	checkCUDNN(cudnnCreateTensorDescriptor(&fcTensor));
	checkCUDNN(cudnnCreateTensorDescriptor(&bias_fcTensor));
	checkCUDNN(cudnnCreateFilterDescriptor(&convFilter_Desc));
	checkCUDNN(cudnnCreateFilterDescriptor(&fcFilter_Desc));
	checkCUDNN(cudnnCreateConvolutionDescriptor(&conv_Desc));
	checkCUDNN(cudnnCreateConvolutionDescriptor(&fc_Desc));
	checkCUDNN(cudnnCreateActivationDescriptor(&convAct_Desc));
	checkCUDNN(cudnnCreatePoolingDescriptor(&pool_Desc));



	/**********FEEDFORWARD********/

	checkCUDNN(cudnnSetTensor4dDescriptor(input_Tensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batchSize, input_channelCnt, imageHeight, imageWidth));

	checkCUDNN(cudnnSetFilter4dDescriptor(convFilter_Desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, filterCnt, input_channelCnt, convFilterHeight, convFilterWidth));

	checkCUDNN(cudnnSetConvolution2dDescriptor(conv_Desc, convPad_h, convPad_w, convStr_h, convStr_w, convDil_h, convDil_w, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

	int convOut_n, convOut_c, convOut_h, convOut_w;
	checkCUDNN(cudnnGetConvolution2dForwardOutputDim(conv_Desc, input_Tensor, convFilter_Desc, &convOut_n, &convOut_c, &convOut_h, &convOut_w));
	checkCUDNN(cudnnSetTensor4dDescriptor(conv_Tensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, convOut_n, convOut_c, convOut_h, convOut_w));

	cudnnConvolutionFwdAlgo_t convFwd_algo;
	checkCUDNN(cudnnGetConvolutionForwardAlgorithm(cudnn, input_Tensor, convFilter_Desc, conv_Desc, conv_Tensor, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &convFwd_algo));
	size_t convWorks_size = 0;
	checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn, input_Tensor, convFilter_Desc, conv_Desc, conv_Tensor, convFwd_algo, &convWorks_size));
	size_t * dev_convWorks;
	checkCudaErrors(cudaMalloc((void**)&dev_convWorks, convWorks_size));
	checkCUDNN(cudnnConvolutionForward(cudnn, &alpha, input_Tensor, dev_input, convFilter_Desc, dev_convFilter, conv_Desc, convFwd_algo, dev_convWorks, convWorks_size, &beta, conv_Tensor, dev_convOutput));


	vector<float> firm_convOutput;
	firm_convOutput.resize(batchSize * filterCnt * convOutHeight * convOutWidth);
	cudaMemcpy(&firm_convOutput[0], dev_convOutput, sizeof(float) * batchSize * filterCnt * convOutHeight * convOutWidth, cudaMemcpyDeviceToHost);


	for (int ch = 0; ch < batchSize * filterCnt * convOutHeight * convOutWidth; ch++)
	{
		if (ch % 32 == 0)
			cout << endl;
		cout << setw(10) << firm_convOutput[ch] << "::";

	}
	cout << endl << "convOut" << endl;



	checkCUDNN(cudnnSetTensor4dDescriptor(bias_convTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, filterCnt, 1, 1));
	checkCUDNN(cudnnAddTensor(cudnn, &alpha, bias_convTensor, dev_convBias, &alpha, conv_Tensor, dev_convOutput));

	checkCUDNN(cudnnSetActivationDescriptor(convAct_Desc, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0));
	checkCUDNN(cudnnActivationForward(cudnn, convAct_Desc, &alpha, conv_Tensor, dev_convOutput, &beta, conv_Tensor, dev_output_Act));

	vector<float> firm_output_Act;
	firm_output_Act.resize(batchSize * filterCnt * convOutHeight * convOutWidth);
	cudaMemcpy(&firm_output_Act[0], dev_output_Act, sizeof(float) * batchSize * filterCnt * convOutHeight * convOutWidth, cudaMemcpyDeviceToHost);


	for (int ch = 0; ch < batchSize * filterCnt * convOutHeight * convOutWidth; ch++)
	{
		if (ch % 32 == 0)
			cout << endl;
		cout << setw(10) << firm_output_Act[ch] << "::";

	}
	cout << endl << "actOut" << endl;



	checkCUDNN(cudnnSetPooling2dDescriptor(pool_Desc, CUDNN_POOLING_MAX, CUDNN_PROPAGATE_NAN, poolWind_h, poolWind_w, poolPad_h, poolPad_w, poolStrd_h, poolStrd_w));

	int poolOut_n, poolOut_c, poolOut_h, poolOut_w;
	checkCUDNN(cudnnGetPooling2dForwardOutputDim(pool_Desc, conv_Tensor, &poolOut_n, &poolOut_c, &poolOut_h, &poolOut_w));
	checkCUDNN(cudnnSetTensor4dDescriptor(poolOutTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, poolOut_n, poolOut_c, poolOut_h, poolOut_w));

	checkCUDNN(cudnnPoolingForward(cudnn, pool_Desc, &alpha, conv_Tensor, dev_output_Act, &beta, poolOutTensor, dev_poolOutput));

	vector<float> firm_poolOutput;
	firm_poolOutput.resize(batchSize * filterCnt * poolOutHeight * poolOutWidth);
	cudaMemcpy(&firm_poolOutput[0], dev_poolOutput, sizeof(float) * batchSize * filterCnt * poolOutHeight * poolOutWidth, cudaMemcpyDeviceToHost);


	for (int ch = 0; ch < batchSize * filterCnt * poolOutHeight * poolOutWidth; ch++)
	{
		if (ch % 16 == 0)
			cout << endl;
		cout << setw(10) << firm_poolOutput[ch] << "::";

	}
	cout << endl << "poolOut" << endl;




	checkCUDNN(cudnnSetFilter4dDescriptor(fcFilter_Desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, lastLayer_numOut, filterCnt, poolOutHeight, poolOutWidth));
	checkCUDNN(cudnnSetConvolution2dDescriptor(fc_Desc, fcPad_h, fcPad_w, fcStr_h, fcStr_w, fcDil_h, fcDil_w, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

	int fwdOut_n, fwdOut_c, fwdOut_h, fwdOut_w;
	checkCUDNN(cudnnGetConvolution2dForwardOutputDim(fc_Desc, poolOutTensor, fcFilter_Desc, &fwdOut_n, &fwdOut_c, &fwdOut_h, &fwdOut_w));
	checkCUDNN(cudnnSetTensor4dDescriptor(fcTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, fwdOut_n, fwdOut_c, fwdOut_h, fwdOut_w));

	cudnnConvolutionFwdAlgo_t fcFwd_algo;
	checkCUDNN(cudnnGetConvolutionForwardAlgorithm(cudnn, poolOutTensor, fcFilter_Desc, fc_Desc, fcTensor, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &fcFwd_algo));
	size_t fcWorks_size = 0;
	checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn, poolOutTensor, fcFilter_Desc, fc_Desc, fcTensor, fcFwd_algo, &fcWorks_size));
	size_t * dev_fcWorks;
	checkCudaErrors(cudaMalloc((void**)&dev_fcWorks, fcWorks_size));
	checkCUDNN(cudnnConvolutionForward(cudnn, &alpha, poolOutTensor, dev_poolOutput, fcFilter_Desc, dev_fcFilter, fc_Desc, fcFwd_algo, dev_fcWorks, fcWorks_size, &beta, fcTensor, dev_fcOutput));

	vector<float> firm_fcOutput;
	firm_fcOutput.resize(batchSize * lastLayer_numOut);
	cudaMemcpy(&firm_fcOutput[0], dev_fcOutput, sizeof(float) * batchSize * lastLayer_numOut * 1 * 1, cudaMemcpyDeviceToHost);


	for (int ch = 0; ch < batchSize * lastLayer_numOut; ch++)
	{

		cout << setw(10) << firm_fcOutput[ch] << "::";

	}

	cout << endl << "fcOut" << endl;




	checkCUDNN(cudnnSetTensor4dDescriptor(bias_fcTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, lastLayer_numOut, 1, 1));
	checkCUDNN(cudnnAddTensor(cudnn, &alpha, bias_fcTensor, dev_fcBias, &alpha, fcTensor, dev_fcOutput));


	checkCUDNN(cudnnSoftmaxForward(cudnn, CUDNN_SOFTMAX_FAST, CUDNN_SOFTMAX_MODE_INSTANCE, &alpha, fcTensor, dev_fcOutput, &beta, fcTensor, dev_smaxOutput));


	vector<float> firm_smaxOutput;
	firm_smaxOutput.resize(batchSize * lastLayer_numOut);
	cudaMemcpy(&firm_smaxOutput[0], dev_smaxOutput, sizeof(float) * batchSize * lastLayer_numOut, cudaMemcpyDeviceToHost);


	for (int ch = 0; ch < batchSize * lastLayer_numOut; ch++)
	{

		cout << setw(10) << firm_smaxOutput[ch] << "::";

	}

	cout << endl << "smaxOut" << endl;



	float *dloss_data;
	cudaMalloc((void**)&dloss_data, sizeof(float) * batchSize * lastLayer_numOut * 1 * 1);


	// 변수 복사
	cudaMemcpy(dloss_data, dev_smaxOutput, sizeof(float) * batchSize * lastLayer_numOut * 1 * 1, cudaMemcpyDeviceToDevice);


	// 라벨, 라벨수, 이미지 총수, 소프트맥스 결과 
	SoftmaxLossBackprop << < RoundUp(100, BW), BW >> > (d_labels, 10, batchSize, dloss_data);


	vector<float> firm_dloss_data;
	firm_dloss_data.resize(batchSize * lastLayer_numOut);
	cudaMemcpy(&firm_dloss_data[0], dloss_data, sizeof(float) * batchSize * lastLayer_numOut, cudaMemcpyDeviceToHost);


	for (int ch = 0; ch < batchSize * lastLayer_numOut; ch++)
	{
		if (ch % 8 == 0)
			cout << endl;
		cout << setw(10) << firm_dloss_data[ch] << "::";

	}

	cout << endl << "dlossOut" << endl;



	/**********BACKPWARD**********/

	float* dev_bw_fcBias;
	checkCudaErrors(cudaMalloc((void**)&dev_bw_fcBias, sizeof(float) * 1 * lastLayer_numOut * 1 * 1));

	float* dev_bwf_fcOutput;
	checkCudaErrors(cudaMalloc((void**)&dev_bwf_fcOutput, sizeof(float) * lastLayer_numOut * filterCnt * poolOutHeight * poolOutWidth));

	float* dev_bwd_fcOutput;
	checkCudaErrors(cudaMalloc((void**)&dev_bwd_fcOutput, sizeof(float) * batchSize * filterCnt * poolOutHeight * poolOutWidth));

	float* dev_poolDelta;
	checkCudaErrors(cudaMalloc((void**)&dev_poolDelta, sizeof(float) * batchSize * filterCnt * convOutHeight * convOutWidth));

	float* dev_bw_actDelta;
	checkCudaErrors(cudaMalloc((void**)&dev_bw_actDelta, sizeof(float) * batchSize * filterCnt * convOutHeight * convOutWidth));

	float* dev_bw_convBias;
	checkCudaErrors(cudaMalloc((void**)&dev_bw_convBias, sizeof(float) * 1 * filterCnt * 1 * 1));

	float* dev_bwf_convOutput;
	checkCudaErrors(cudaMalloc((void**)&dev_bwf_convOutput, sizeof(float) * filterCnt * input_channelCnt * convFilterHeight * convFilterWidth));





	cudnnConvolutionBwdFilterAlgo_t bwf_fcAlgo;
	checkCUDNN(cudnnGetConvolutionBackwardFilterAlgorithm(cudnn, poolOutTensor, fcTensor, fc_Desc, fcFilter_Desc, CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, &bwf_fcAlgo));
	size_t fcBackFilterWorks_size = 0;
	checkCUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnn, poolOutTensor, fcTensor, fc_Desc, fcFilter_Desc, bwf_fcAlgo, &fcBackFilterWorks_size));
	size_t* dev_fcBackFilterWorks;
	checkCudaErrors(cudaMalloc((void**)&dev_fcBackFilterWorks, fcBackFilterWorks_size));

	cudnnConvolutionBwdDataAlgo_t bwd_fcAlgo;
	checkCUDNN(cudnnGetConvolutionBackwardDataAlgorithm(cudnn, fcFilter_Desc, fcTensor, fc_Desc, poolOutTensor, CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, &bwd_fcAlgo));
	size_t fcBackDataWorks_size = 0;
	checkCUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn, fcFilter_Desc, fcTensor, fc_Desc, poolOutTensor, bwd_fcAlgo, &fcBackDataWorks_size));
	size_t* dev_fcBackDataWorks;
	checkCudaErrors(cudaMalloc((void**)&dev_fcBackDataWorks, fcBackDataWorks_size));

	cudnnConvolutionBwdFilterAlgo_t bwf_convAlgo;
	checkCUDNN(cudnnGetConvolutionBackwardFilterAlgorithm(cudnn, input_Tensor, conv_Tensor, conv_Desc, convFilter_Desc, CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, &bwf_convAlgo));
	size_t convBackFilterWorks_size = 0;
	checkCUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnn, input_Tensor, conv_Tensor, conv_Desc, convFilter_Desc, bwf_convAlgo, &convBackFilterWorks_size));
	size_t * dev_convBackDataWorks;
	checkCudaErrors(cudaMalloc((void**)&dev_convBackDataWorks, convBackFilterWorks_size));


	/*
	float* diffData; // 계산하기
	float* dev_diffData;
	checkCudaErrors(cudaMalloc((void**)&dev_diffData, sizeof(diffData)));
	checkCudaErrors(cudaMemcpy(dev_diffData, diffData, sizeof(diffData), cudaMemcpyHostToDevice));
	checkCUDNN(cudnnSoftmaxBackward(cudnn, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, &alpha, fcTensor, dev_smaxOutput, diffTensorDesc, dev_diffData, &beta, fcTensor, dev_bw_smaxData));
	*/


	checkCUDNN(cudnnConvolutionBackwardBias(cudnn, &alpha, fcTensor, dloss_data, &beta, bias_fcTensor, dev_bw_fcBias));

	checkCUDNN(cudnnConvolutionBackwardFilter(cudnn, &alpha, poolOutTensor, dev_poolOutput, fcTensor, dloss_data, fc_Desc,
		bwf_fcAlgo, dev_fcBackFilterWorks, fcBackFilterWorks_size, &beta, fcFilter_Desc, dev_bwf_fcOutput));

	checkCUDNN(cudnnConvolutionBackwardData(cudnn, &alpha, fcFilter_Desc, dev_fcFilter, fcTensor, dloss_data,
		fc_Desc, bwd_fcAlgo, dev_fcBackDataWorks, fcBackDataWorks_size, &beta, poolOutTensor, dev_bwd_fcOutput));

	//**********************************
	//**DropOut Backpropagtion**         추가사항
	//**********************************

	checkCUDNN(cudnnPoolingBackward(cudnn, pool_Desc, &alpha, poolOutTensor, dev_poolOutput, poolOutTensor,
		dev_bwd_fcOutput, conv_Tensor, dev_convOutput, &beta, conv_Tensor, dev_poolDelta));

	checkCUDNN(cudnnActivationBackward(cudnn, convAct_Desc, &alpha, conv_Tensor, dev_output_Act, conv_Tensor,
		dev_poolDelta, conv_Tensor, dev_convOutput, &beta, conv_Tensor, dev_bw_actDelta));

	checkCUDNN(cudnnConvolutionBackwardBias(cudnn, &alpha, conv_Tensor, dev_bw_actDelta, &beta, bias_convTensor, dev_bw_convBias));

	checkCUDNN(cudnnConvolutionBackwardFilter(cudnn, &alpha, input_Tensor, dev_input, conv_Tensor,
		dev_bw_actDelta, conv_Desc, bwf_convAlgo, dev_convBackDataWorks,
		convBackFilterWorks_size, &beta, convFilter_Desc, dev_bwf_convOutput));



	//weight update
	float learning_rate = -0.01;
	// Conv1
	cublasSaxpy(cublasHandle, static_cast<int>(filterCnt * input_channelCnt * convFilterHeight * convFilterWidth), &learning_rate, dev_bwf_convOutput, 1, dev_convFilter, 1);
	cublasSaxpy(cublasHandle, static_cast<int>(1 * filterCnt * 1 * 1), &learning_rate, dev_bw_convBias, 1, dev_convBias, 1);

	// Fully connected 1
	cublasSaxpy(cublasHandle, static_cast<int>(lastLayer_numOut * filterCnt * poolOutHeight * poolOutWidth), &learning_rate, dev_bwf_fcOutput, 1, dev_fcFilter, 1);
	cublasSaxpy(cublasHandle, static_cast<int>(1 * lastLayer_numOut * 1 * 1), &learning_rate, dev_bw_fcBias, 1, dev_fcBias, 1);



	std::vector<float> Filter2; //[filterCnt][input_channelCnt][convFilterHeight][convFilterWidth]
	Filter2.resize(filterCnt * input_channelCnt * convFilterHeight * convFilterWidth);

	cudaMemcpy(&Filter2[0], dev_convFilter, sizeof(float) * filterCnt * input_channelCnt * convFilterHeight * convFilterWidth, cudaMemcpyDeviceToHost);


	for (int ch = 0; ch < 5 * 3 * 3 * 3; ch++)
	{
		if (ch % 8 == 0)
			cout << endl;
		cout << setw(10) << Filter2[ch] << "::";

	}

	std::cout << "끝==========================================" << endl;

	cudaFree(dev_bwf_convOutput);
	cudaFree(dev_bw_convBias);
	cudaFree(dev_bw_actDelta);
	cudaFree(dev_poolDelta);
	cudaFree(dev_bwd_fcOutput);

	cudaFree(dev_bwf_fcOutput);
	cudaFree(dev_bw_fcBias);


	cudaFree(dev_smaxOutput);
	cudaFree(dev_fcBias);

	cudaFree(dev_fcOutput);
	cudaFree(dev_fcFilter);
	cudaFree(dev_poolOutput);
	cudaFree(dev_output_Act);
	cudaFree(dev_convBias);

	cudaFree(dev_convOutput);
	cudaFree(dev_convFilter);
	cudaFree(dev_input);





}