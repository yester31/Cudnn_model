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

#define BW 512

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

void InitWeightsXavier(float* Weights,
	const size_t NextLayerNodeNumber/*Number of output  feature maps*/,
	const size_t PrevLayerNodeNumber/*Number of input feature maps*/,
	const size_t Height/*Height of each filter.*/,
	const size_t Width/*Width of each filter.*/)
{

	random_device rd;
	mt19937 gen(rd());
	float sigma = sqrt(6.0f / static_cast<float>((NextLayerNodeNumber + PrevLayerNodeNumber) * Height * Width));
	uniform_real_distribution<float> d(-sigma, sigma);

	//Weights 정의
	for (int och = 0; och < NextLayerNodeNumber; och++)
	{
		for (int ch = 0; ch < PrevLayerNodeNumber; ch++)
		{
			for (int row = 0; row < Height; row++)
			{
				for (int col = 0; col < Width; col++)
				{
					Weights[och * PrevLayerNodeNumber * Height * Width + ch *  Height * Width + row * Width + col] = static_cast<float>(d(gen));
				}
			}
		}
	}

}


void InitWeightsXavierC3(float* Weights,
	const size_t NextLayerNodeNumber/*Number of output  feature maps*/,
	const size_t PrevLayerNodeNumber/*Number of input feature maps*/,
	const size_t Height/*Height of each filter.*/,
	const size_t Width/*Width of each filter.*/)
{

	random_device rd;
	mt19937 gen(rd());
	float sigma = sqrt(6.0f / static_cast<float>((NextLayerNodeNumber + PrevLayerNodeNumber) * Height * Width));
	uniform_real_distribution<float> d(-sigma, sigma);

	//Weights 정의
	for (int och = 0; och < NextLayerNodeNumber; och++)
	{
		for (int ch = 0; ch < PrevLayerNodeNumber; ch++)
		{
			for (int row = 0; row < Height; row++)
			{
				for (int col = 0; col < Width; col++)
				{
					Weights[och * PrevLayerNodeNumber * Height * Width + ch *  Height * Width + row * Width + col] = static_cast<float>(d(gen));
				}
			}
		}
	}

}


void InitWeightsbias(float* Weightsbias, const size_t & numOutSize)
{
	for (int i = 0; i < numOutSize; i++)
	{
		Weightsbias[i] = 0.0f;
	}
}






int main()
{
	time_t startTime = 0, endTime = 0;
	time_t startTime_train = 0, endTime_train = 0;

	startTime = clock();

	const int num_labels = 10; // 라벨 수

	int ImageNum = 50000; // 이미지 총 개수
	int batchSize = 100; // 트레이닝 배치 수

	int ImageNum_test = 10000; // 이미지 총 개수
	const int batch_size_test = 100; // 테스트 배치 수

	int epoch = 50; // epoch 횟수 

	float learning_rate = -0.001; // 학습률


	int input_channelCnt = 3, imageHeight = 224, imageWidth = 224;


	vector<pair<Mat, string>> ImgBox; // 이미지 데이터, 이미지 이름
	ImgBox = TraverseFilesUsingDFS("C:\\cifar\\train");// 이미지가 저장되어 있는 폴더 경로
													   //ImgBox = TraverseFilesUsingDFS("C:\\Users\\ECMUser\\Desktop\\DataSet\\MNIST11000\\trainset");// 이미지가 저장되어 있는 폴더 경로
	vector<string> LabelBox; // 라벨 정리를 위해 생성
	vector<pair<int, string>> LabelTable; // 라벨링 마다 넘버 부여

	vector<pair<Mat, string>> ImgBox_test; // 이미지 데이터, 이미지 이름
	ImgBox_test = TraverseFilesUsingDFS("C:\\cifar\\test");// 이미지가 저장되어 있는 폴더 경로
														   //ImgBox_test = TraverseFilesUsingDFS("C:\\Users\\ECMUser\\Desktop\\DataSet\\cifar_10\\test");// 이미지가 저장되어 있는 폴더 경로
														   //ImgBox_test = TraverseFilesUsingDFS("C:\\Users\\ECMUser\\Desktop\\DataSet\\MNIST11000\\testset");// 이미지가 저장되어 있는 폴더 경로

	float* target_train = new float[ImageNum]; // target 값 , 라벨에 따른 지정된 넘버 값이 담긴 배열 

											   // 라벨에 번호 부여를 위해 LabelBox 벡터에 값 복사 하고 정렬 및 중복 삭제
	for (int i = 0; i < ImageNum; i++)
	{
		//std::cout<< "라벨 출력 :: " << ImgBox[i].second << std::endl; // 입력받은순서대로 라벨 출력 -> 예시 "라벨 출력 :: automobile"
		LabelBox.push_back(ImgBox[i].second);
	}

	sort(LabelBox.begin(), LabelBox.end());
	LabelBox.erase(unique(LabelBox.begin(), LabelBox.end()), LabelBox.end());
	int nLabelBoxSize = LabelBox.size();

	// 라벨 번호 부여
	for (int i = 0; i < nLabelBoxSize; i++)
	{
		LabelTable.push_back({ { i },{ LabelBox[i] } });
		//std::cout << "LabelBox :: " << LabelBox[i] << std::endl;// -> 예시 "LabelBox :: truck"
	}

	//target 셋팅
	for (int i = 0; i < ImageNum; i++) {
		for (int j = 0; j < LabelTable.size(); j++) {
			if (ImgBox[i].second == LabelTable[j].second) {
				target_train[i] = LabelTable[j].first;
			}
		}
	}


	cout << "=================== 이미지 불러오기 성공 ====================" << endl;
	endTime = clock();
	printf("이미지 준비 시간: %.1f 분 \n", (float)(endTime - startTime) / ((CLOCKS_PER_SEC) * 60));

	cudnnHandle_t cudnn;
	cudnnCreate(&cudnn);
	cublasHandle_t cublasHandle;
	cublasCreate(&cublasHandle);

	float alpha = 1.0;
	float beta = 0.0;


	float* Input_train = new float[batchSize * input_channelCnt * imageHeight * imageWidth];
	float* target_train_batch = new float[batchSize];

	// filater 정보 

	const int conv1FilterCnt = 96, conv1FilterHeight = 7, conv1FilterWidth = 7;
	int conv1Pad_h = 0, conv1Pad_w = 0, conv1Str_h = 2, conv1Str_w = 2, conv1Dil_h = 1, conv1Dil_w = 1;

	int conv2FilterCnt = 256, conv2FilterHeight = 5, conv2FilterWidth = 5;
	int conv2Pad_h = 0, conv2Pad_w = 0, conv2Str_h = 2, conv2Str_w =2, conv2Dil_h = 1, conv2Dil_w = 1;

	int conv3FilterCnt = 384, conv3FilterHeight = 3, conv3FilterWidth = 3;
	int conv3Pad_h = 0, conv3Pad_w = 0, conv3Str_h = 1, conv3Str_w = 1, conv3Dil_h = 1, conv3Dil_w = 1;

	int conv4FilterCnt = 384, conv4FilterHeight = 3, conv4FilterWidth = 3;
	int conv4Pad_h = 0, conv4Pad_w = 0, conv4Str_h = 1, conv4Str_w = 1, conv4Dil_h = 1, conv4Dil_w = 1;

	int conv5FilterCnt = 256, conv5FilterHeight = 3, conv5FilterWidth = 3;
	int conv5Pad_h = 0, conv5Pad_w = 0, conv5Str_h = 1, conv5Str_w = 1, conv5Dil_h = 1, conv5Dil_w = 1;

	// pooling 정보
	int poolWind_h = 3, poolWind_w = 3, poolPad_h = 0, poolPad_w = 0, poolStrd_w = 2, poolStrd_h = 2; //MAX POOLING 고정

	//Layer 1
	int conv1OutHeight = 110;
	int conv1OutWidth = 110;

	int pool1OutHeight = 55;
	int pool1OutWidth = 55;

	//Layer 2
	int conv2OutHeight = 26;
	int conv2OutWidth = 26;

	int pool2OutHeight = 13;
	int pool2OutWidth = 13;

	//Layer 3
	int conv3OutHeight = 13;
	int conv3OutWidth = 13;

	//Layer 4
	int conv4OutHeight = 13;
	int conv4OutWidth = 13;

	//Layer 5
	int conv5OutHeight = 13;
	int conv5OutWidth = 13;

	int pool5OutHeight = 6;
	int pool5OutWidth = 6;


	int fcPad_h = 0, fcPad_w = 0, fcStr_h = 1, fcStr_w = 1, fcDil_h = 1, fcDil_w = 1; //얘도 고정

	// Layer 6
	int fc1FilterCnt = 4096; 

	// Layer 7
	int fc2FilterCnt = 4096; 

	int lastLayer_numOut = 1000; // 출력 클래스 수, Output



	//Weight initialization

	//conv1 filter
	float* conv1Filter = new float[conv1FilterCnt * input_channelCnt * conv1FilterHeight * conv1FilterWidth];
	InitWeightsXavier(conv1Filter, conv1FilterCnt, input_channelCnt, conv1FilterHeight, conv1FilterWidth);

	//conv2 filter
	float* conv2Filter = new float[conv2FilterCnt * conv1FilterCnt * conv2FilterHeight * conv2FilterWidth];
	InitWeightsXavier(conv2Filter, conv2FilterCnt, conv1FilterCnt, conv2FilterHeight, conv2FilterWidth);


	//conv3 filter
	float* conv3Filter = new float[conv3FilterCnt * conv2FilterCnt * conv3FilterHeight * conv3FilterWidth];
	InitWeightsXavier(conv3Filter, conv3FilterCnt, conv2FilterCnt, conv3FilterHeight, conv3FilterWidth);

	//conv4 filter
	float* conv4Filter = new float[conv4FilterCnt * conv3FilterCnt * conv4FilterHeight * conv4FilterWidth];
	InitWeightsXavier(conv4Filter, conv4FilterCnt, conv3FilterCnt, conv4FilterHeight, conv4FilterWidth);

	//conv5 filter
	float* conv5Filter = new float[conv5FilterCnt * conv4FilterCnt * conv5FilterHeight * conv5FilterWidth];
	InitWeightsXavier(conv5Filter, conv5FilterCnt, conv4FilterCnt, conv5FilterHeight, conv5FilterWidth);


	//fc1 filter
	float* fc1Filter = new float[fc1FilterCnt * conv5FilterCnt * pool5OutHeight * pool5OutWidth];
	InitWeightsXavier(fc1Filter, fc1FilterCnt, conv5FilterCnt, pool5OutHeight, pool5OutWidth);

	//fc2 filter
	float* fc2Filter = new float[fc2FilterCnt * fc1FilterCnt * 1 * 1];
	InitWeightsXavier(fc2Filter, fc2FilterCnt, fc1FilterCnt, 1, 1);


	// fc3 filter
	float* fc3Filter = new float[lastLayer_numOut * fc2FilterCnt * 1 * 1];
	InitWeightsXavier(fc3Filter, lastLayer_numOut, fc2FilterCnt, 1, 1);




	//conv1 bias
	float* conv1Bias = new float[conv1FilterCnt];
	InitWeightsbias(conv1Bias, conv1FilterCnt);

	// conv2 bias
	float* conv2Bias = new float[conv2FilterCnt];
	InitWeightsbias(conv2Bias, conv2FilterCnt);

	// conv3 bias
	float* conv3Bias = new float[conv3FilterCnt];
	InitWeightsbias(conv3Bias, conv3FilterCnt);

	// conv4 bias
	float* conv4Bias = new float[conv4FilterCnt];
	InitWeightsbias(conv4Bias, conv4FilterCnt);

	// conv2 bias
	float* conv5Bias = new float[conv5FilterCnt];
	InitWeightsbias(conv5Bias, conv5FilterCnt);

	//fc1 bias
	float* fc1Bias = new float[fc1FilterCnt];
	InitWeightsbias(fc1Bias, fc1FilterCnt);

	// fc2 bias
	float* fc2Bias = new float[fc2FilterCnt];
	InitWeightsbias(fc2Bias, fc2FilterCnt);

	// fc3 bias
	float* fc3Bias = new float[lastLayer_numOut];
	InitWeightsbias(fc3Bias, lastLayer_numOut);

	////////////////1

	float* dev_conv1_Filter;
	checkCudaErrors(cudaMalloc((void**)&dev_conv1_Filter, sizeof(float) * conv1FilterCnt * input_channelCnt * conv1FilterHeight * conv1FilterWidth));
	checkCudaErrors(cudaMemcpy(dev_conv1_Filter, conv1Filter, sizeof(float) * conv1FilterCnt * input_channelCnt * conv1FilterHeight * conv1FilterWidth, cudaMemcpyHostToDevice));

	float* dev_conv1_Output;
	checkCudaErrors(cudaMalloc((void**)&dev_conv1_Output, sizeof(float) * batchSize * conv1FilterCnt * conv1OutHeight * conv1OutWidth));

	float* dev_conv1_Bias;
	checkCudaErrors(cudaMalloc((void**)&dev_conv1_Bias, sizeof(float) * 1 * conv1FilterCnt * 1 * 1));
	checkCudaErrors(cudaMemcpy(dev_conv1_Bias, conv1Bias, sizeof(float) * 1 * conv1FilterCnt * 1 * 1, cudaMemcpyHostToDevice));

	float* dev_conv1Act_output;
	checkCudaErrors(cudaMalloc((void**)&dev_conv1Act_output, sizeof(float) * batchSize * conv1FilterCnt * conv1OutHeight * conv1OutWidth));

	float* dev_pool1_Output;
	checkCudaErrors(cudaMalloc((void**)&dev_pool1_Output, sizeof(float) * batchSize * conv1FilterCnt * pool1OutHeight * pool1OutWidth));

	float* dev_conv1pool_Act_output;
	checkCudaErrors(cudaMalloc((void**)&dev_conv1pool_Act_output, sizeof(float) * batchSize * conv1FilterCnt * pool1OutHeight * pool1OutWidth));

	////////////////////2

	float* dev_conv2_Filter;
	checkCudaErrors(cudaMalloc((void**)&dev_conv2_Filter, sizeof(float) * conv2FilterCnt * conv1FilterCnt * conv2FilterHeight * conv2FilterWidth));
	checkCudaErrors(cudaMemcpy(dev_conv2_Filter, conv2Filter, sizeof(float) * conv2FilterCnt * conv1FilterCnt * conv2FilterHeight * conv2FilterWidth, cudaMemcpyHostToDevice));

	float* dev_conv2_Output;
	checkCudaErrors(cudaMalloc((void**)&dev_conv2_Output, sizeof(float) * batchSize * conv2FilterCnt * conv2OutHeight * conv2OutWidth));

	float* dev_conv2_Bias;
	checkCudaErrors(cudaMalloc((void**)&dev_conv2_Bias, sizeof(float) * 1 * conv2FilterCnt * 1 * 1));
	checkCudaErrors(cudaMemcpy(dev_conv2_Bias, conv2Bias, sizeof(float) * 1 * conv2FilterCnt * 1 * 1, cudaMemcpyHostToDevice));

	float* dev_conv2Act_output;
	checkCudaErrors(cudaMalloc((void**)&dev_conv2Act_output, sizeof(float) * batchSize * conv2FilterCnt * conv2OutHeight * conv2OutWidth));

	float* dev_pool2_Output;
	checkCudaErrors(cudaMalloc((void**)&dev_pool2_Output, sizeof(float) * batchSize * conv2FilterCnt * pool2OutHeight * pool2OutWidth));

	float* dev_conv2pool_Act_output;
	checkCudaErrors(cudaMalloc((void**)&dev_conv2pool_Act_output, sizeof(float) * batchSize * conv2FilterCnt * pool2OutHeight * pool2OutWidth));

	///////////////////////3

	float* dev_conv3_Filter;
	checkCudaErrors(cudaMalloc((void**)&dev_conv3_Filter, sizeof(float) * conv3FilterCnt * conv2FilterCnt * conv3FilterHeight * conv3FilterWidth));
	checkCudaErrors(cudaMemcpy(dev_conv3_Filter, conv3Filter, sizeof(float) * conv3FilterCnt * conv2FilterCnt * conv3FilterHeight * conv3FilterWidth, cudaMemcpyHostToDevice));

	float* dev_conv3_Output;
	checkCudaErrors(cudaMalloc((void**)&dev_conv3_Output, sizeof(float) * batchSize * conv3FilterCnt * conv3OutHeight * conv3OutWidth));

	float* dev_conv3_Bias;
	checkCudaErrors(cudaMalloc((void**)&dev_conv3_Bias, sizeof(float) * 1 * conv3FilterCnt * 1 * 1));
	checkCudaErrors(cudaMemcpy(dev_conv3_Bias, conv3Bias, sizeof(float) * 1 * conv3FilterCnt * 1 * 1, cudaMemcpyHostToDevice));

	float* dev_conv3Act_output;
	checkCudaErrors(cudaMalloc((void**)&dev_conv3Act_output, sizeof(float) * batchSize * conv3FilterCnt * conv3OutHeight * conv3OutWidth));


	/////////////////4


	float* dev_conv4_Filter;
	checkCudaErrors(cudaMalloc((void**)&dev_conv4_Filter, sizeof(float) * conv4FilterCnt * conv3FilterCnt * conv4FilterHeight * conv4FilterWidth));
	checkCudaErrors(cudaMemcpy(dev_conv4_Filter, conv4Filter, sizeof(float) * conv4FilterCnt * conv3FilterCnt * conv4FilterHeight * conv4FilterWidth, cudaMemcpyHostToDevice));

	float* dev_conv4_Output;
	checkCudaErrors(cudaMalloc((void**)&dev_conv4_Output, sizeof(float) * batchSize * conv4FilterCnt * conv4OutHeight * conv4OutWidth));

	float* dev_conv4_Bias;
	checkCudaErrors(cudaMalloc((void**)&dev_conv4_Bias, sizeof(float) * 1 * conv4FilterCnt * 1 * 1));
	checkCudaErrors(cudaMemcpy(dev_conv4_Bias, conv4Bias, sizeof(float) * 1 * conv4FilterCnt * 1 * 1, cudaMemcpyHostToDevice));

	float* dev_conv4Act_output;
	checkCudaErrors(cudaMalloc((void**)&dev_conv4Act_output, sizeof(float) * batchSize * conv4FilterCnt * conv4OutHeight * conv4OutWidth));


	/////////////////5

	float* dev_conv5_Filter;
	checkCudaErrors(cudaMalloc((void**)&dev_conv5_Filter, sizeof(float) * conv5FilterCnt * conv4FilterCnt * conv5FilterHeight * conv5FilterWidth));
	checkCudaErrors(cudaMemcpy(dev_conv5_Filter, conv5Filter, sizeof(float) * conv5FilterCnt * conv4FilterCnt * conv5FilterHeight * conv5FilterWidth, cudaMemcpyHostToDevice));

	float* dev_conv5_Output;
	checkCudaErrors(cudaMalloc((void**)&dev_conv5_Output, sizeof(float) * batchSize * conv5FilterCnt * conv5OutHeight * conv5OutWidth));

	float* dev_conv5_Bias;
	checkCudaErrors(cudaMalloc((void**)&dev_conv5_Bias, sizeof(float) * 1 * conv5FilterCnt * 1 * 1));
	checkCudaErrors(cudaMemcpy(dev_conv5_Bias, conv5Bias, sizeof(float) * 1 * conv5FilterCnt * 1 * 1, cudaMemcpyHostToDevice));

	float* dev_conv5Act_output;
	checkCudaErrors(cudaMalloc((void**)&dev_conv5Act_output, sizeof(float) * batchSize * conv5FilterCnt * conv5OutHeight * conv5OutWidth));

	float* dev_pool5_Output;
	checkCudaErrors(cudaMalloc((void**)&dev_pool5_Output, sizeof(float) * batchSize * conv5FilterCnt * pool5OutHeight * pool5OutWidth));

	float* dev_conv5pool_Act_output;
	checkCudaErrors(cudaMalloc((void**)&dev_conv5pool_Act_output, sizeof(float) * batchSize * conv5FilterCnt * pool5OutHeight * pool5OutWidth));






	float* dev_fc1_Filter;
	checkCudaErrors(cudaMalloc((void**)&dev_fc1_Filter, sizeof(float) * fc1FilterCnt * conv5FilterCnt * pool2OutHeight * pool2OutWidth));
	checkCudaErrors(cudaMemcpy(dev_fc1_Filter, fc1Filter, sizeof(float) * fc1FilterCnt * conv5FilterCnt * pool2OutHeight * pool2OutWidth, cudaMemcpyHostToDevice));

	float* dev_fc1_Output;
	checkCudaErrors(cudaMalloc((void**)&dev_fc1_Output, sizeof(float) * batchSize * fc1FilterCnt * 1 * 1));

	float* dev_fc1_Bias;
	checkCudaErrors(cudaMalloc((void**)&dev_fc1_Bias, sizeof(float) * 1 * fc1FilterCnt * 1 * 1));
	checkCudaErrors(cudaMemcpy(dev_fc1_Bias, fc1Bias, sizeof(float) * 1 * fc1FilterCnt * 1 * 1, cudaMemcpyHostToDevice));

	float* dev_fc1_Actout;
	checkCudaErrors(cudaMalloc((void**)&dev_fc1_Actout, sizeof(float) * batchSize * fc1FilterCnt * 1 * 1));

	float* dev_fc2_Filter;
	checkCudaErrors(cudaMalloc((void**)&dev_fc2_Filter, sizeof(float) * fc2FilterCnt * fc1FilterCnt * 1 * 1));
	checkCudaErrors(cudaMemcpy(dev_fc2_Filter, fc2Filter, sizeof(float) * fc2FilterCnt * fc1FilterCnt * 1 * 1, cudaMemcpyHostToDevice));

	float* dev_fc2_Output;
	checkCudaErrors(cudaMalloc((void**)&dev_fc2_Output, sizeof(float) * batchSize * fc2FilterCnt * 1 * 1));

	float* dev_fc2_Bias;
	checkCudaErrors(cudaMalloc((void**)&dev_fc2_Bias, sizeof(float) * 1 * fc2FilterCnt * 1 * 1));
	checkCudaErrors(cudaMemcpy(dev_fc2_Bias, fc2Bias, sizeof(float) * 1 * fc2FilterCnt * 1 * 1, cudaMemcpyHostToDevice));

	float* dev_fc2_Actout;
	checkCudaErrors(cudaMalloc((void**)&dev_fc2_Actout, sizeof(float) * batchSize * fc2FilterCnt * 1 * 1));

	float* dev_fc3_Filter;
	checkCudaErrors(cudaMalloc((void**)&dev_fc3_Filter, sizeof(float) * lastLayer_numOut * fc2FilterCnt * 1 * 1));
	checkCudaErrors(cudaMemcpy(dev_fc3_Filter, fc3Filter, sizeof(float) * lastLayer_numOut * fc2FilterCnt * 1 * 1, cudaMemcpyHostToDevice));

	float* dev_fc3_Output;
	checkCudaErrors(cudaMalloc((void**)&dev_fc3_Output, sizeof(float) * batchSize * lastLayer_numOut * 1 * 1));

	float* dev_fc3_Bias;
	checkCudaErrors(cudaMalloc((void**)&dev_fc3_Bias, sizeof(float) * 1 * lastLayer_numOut * 1 * 1));
	checkCudaErrors(cudaMemcpy(dev_fc3_Bias, fc3Bias, sizeof(float) * 1 * lastLayer_numOut * 1 * 1, cudaMemcpyHostToDevice));

	float* dev_smaxOutput;
	checkCudaErrors(cudaMalloc((void**)&dev_smaxOutput, sizeof(float) * batchSize * lastLayer_numOut * 1 * 1));



	/**********FEEDFORWARD********/

	//cudnnPoolingMode_t poolMode = CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
	cudnnPoolingMode_t poolMode = CUDNN_POOLING_MAX;

	cudnnActivationMode_t fwActalgo = CUDNN_ACTIVATION_RELU;
	//cudnnActivationMode_t fwActalgo = CUDNN_ACTIVATION_TANH;
	//cudnnActivationMode_t fwActalgo = CUDNN_ACTIVATION_SIGMOID;

	//cudnnSoftmaxAlgorithm_t sftAlgo = CUDNN_SOFTMAX_FAST;
	cudnnSoftmaxAlgorithm_t sftAlgo = CUDNN_SOFTMAX_ACCURATE;
	cudnnSoftmaxMode_t sftMode = CUDNN_SOFTMAX_MODE_INSTANCE;
	//cudnnSoftmaxMode_t sftMode = CUDNN_SOFTMAX_MODE_CHANNEL;


	cudnnTensorDescriptor_t input_Tensor;
	checkCUDNN(cudnnCreateTensorDescriptor(&input_Tensor));
	checkCUDNN(cudnnSetTensor4dDescriptor(input_Tensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batchSize, input_channelCnt, imageHeight, imageWidth));

	cudnnFilterDescriptor_t conv1_Filter_Desc;
	checkCUDNN(cudnnCreateFilterDescriptor(&conv1_Filter_Desc));
	checkCUDNN(cudnnSetFilter4dDescriptor(conv1_Filter_Desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, conv1FilterCnt, input_channelCnt, conv1FilterHeight, conv1FilterWidth));

	cudnnConvolutionDescriptor_t conv1_Desc;
	checkCUDNN(cudnnCreateConvolutionDescriptor(&conv1_Desc));
	checkCUDNN(cudnnSetConvolution2dDescriptor(conv1_Desc, conv1Pad_h, conv1Pad_w, conv1Str_h, conv1Str_w, conv1Dil_h, conv1Dil_w, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

	cudnnTensorDescriptor_t conv1_Tensor;
	checkCUDNN(cudnnCreateTensorDescriptor(&conv1_Tensor));
	int conv1Out_n, conv1Out_c, conv1Out_h, conv1Out_w;
	checkCUDNN(cudnnGetConvolution2dForwardOutputDim(conv1_Desc, input_Tensor, conv1_Filter_Desc, &conv1Out_n, &conv1Out_c, &conv1Out_h, &conv1Out_w));
	checkCUDNN(cudnnSetTensor4dDescriptor(conv1_Tensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, conv1Out_n, conv1Out_c, conv1Out_h, conv1Out_w));

	cudnnConvolutionFwdAlgo_t conv1_fwAlgo;
	checkCUDNN(cudnnGetConvolutionForwardAlgorithm(cudnn, input_Tensor, conv1_Filter_Desc, conv1_Desc, conv1_Tensor, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &conv1_fwAlgo));
	size_t conv1_worksSize = 0;
	checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn, input_Tensor, conv1_Filter_Desc, conv1_Desc, conv1_Tensor, conv1_fwAlgo, &conv1_worksSize));
	size_t * dev_conv1_works;
	checkCudaErrors(cudaMalloc((void**)&dev_conv1_works, conv1_worksSize));

	cudnnTensorDescriptor_t conv1_biasTensor;
	checkCUDNN(cudnnCreateTensorDescriptor(&conv1_biasTensor));
	checkCUDNN(cudnnSetTensor4dDescriptor(conv1_biasTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, conv1FilterCnt, 1, 1));

	cudnnActivationDescriptor_t conv1_Act_Desc;
	checkCUDNN(cudnnCreateActivationDescriptor(&conv1_Act_Desc));
	checkCUDNN(cudnnSetActivationDescriptor(conv1_Act_Desc, fwActalgo, CUDNN_PROPAGATE_NAN, 0));

	cudnnPoolingDescriptor_t pool1_Desc;
	checkCUDNN(cudnnCreatePoolingDescriptor(&pool1_Desc));
	checkCUDNN(cudnnSetPooling2dDescriptor(pool1_Desc, poolMode, CUDNN_PROPAGATE_NAN, poolWind_h, poolWind_w, poolPad_h, poolPad_w, poolStrd_h, poolStrd_w));

	cudnnTensorDescriptor_t pool1_Tensor;
	checkCUDNN(cudnnCreateTensorDescriptor(&pool1_Tensor));
	int pool1Out_n, pool1Out_c, pool1Out_h, pool1Out_w;
	checkCUDNN(cudnnGetPooling2dForwardOutputDim(pool1_Desc, conv1_Tensor, &pool1Out_n, &pool1Out_c, &pool1Out_h, &pool1Out_w));
	checkCUDNN(cudnnSetTensor4dDescriptor(pool1_Tensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, pool1Out_n, pool1Out_c, pool1Out_h, pool1Out_w));

	cudnnActivationDescriptor_t conv1pool_Act_Desc;
	checkCUDNN(cudnnCreateActivationDescriptor(&conv1pool_Act_Desc));
	checkCUDNN(cudnnSetActivationDescriptor(conv1pool_Act_Desc, fwActalgo, CUDNN_PROPAGATE_NAN, 0));

	cudnnFilterDescriptor_t conv2_Filter_Desc;
	checkCUDNN(cudnnCreateFilterDescriptor(&conv2_Filter_Desc));
	checkCUDNN(cudnnSetFilter4dDescriptor(conv2_Filter_Desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, conv2FilterCnt, conv1FilterCnt, conv2FilterHeight, conv2FilterWidth));

	cudnnConvolutionDescriptor_t conv2_Desc;
	checkCUDNN(cudnnCreateConvolutionDescriptor(&conv2_Desc));
	checkCUDNN(cudnnSetConvolution2dDescriptor(conv2_Desc, conv2Pad_h, conv2Pad_w, conv2Str_h, conv2Str_w, conv2Dil_h, conv2Dil_w, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

	cudnnTensorDescriptor_t conv2_Tensor;
	checkCUDNN(cudnnCreateTensorDescriptor(&conv2_Tensor));
	int conv2Out_n, conv2Out_c, conv2Out_h, conv2Out_w;
	checkCUDNN(cudnnGetConvolution2dForwardOutputDim(conv2_Desc, pool1_Tensor, conv2_Filter_Desc, &conv2Out_n, &conv2Out_c, &conv2Out_h, &conv2Out_w));
	checkCUDNN(cudnnSetTensor4dDescriptor(conv2_Tensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, conv2Out_n, conv2Out_c, conv2Out_h, conv2Out_w));

	cudnnConvolutionFwdAlgo_t conv2_fwAlgo;
	checkCUDNN(cudnnGetConvolutionForwardAlgorithm(cudnn, pool1_Tensor, conv2_Filter_Desc, conv2_Desc, conv2_Tensor, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &conv2_fwAlgo));
	size_t conv2_worksSize = 0;
	checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn, pool1_Tensor, conv2_Filter_Desc, conv2_Desc, conv2_Tensor, conv2_fwAlgo, &conv2_worksSize));
	size_t * dev_conv2_works;
	checkCudaErrors(cudaMalloc((void**)&dev_conv2_works, conv2_worksSize));

	cudnnTensorDescriptor_t conv2_biasTensor;
	checkCUDNN(cudnnCreateTensorDescriptor(&conv2_biasTensor));
	checkCUDNN(cudnnSetTensor4dDescriptor(conv2_biasTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, conv2FilterCnt, 1, 1));

	cudnnActivationDescriptor_t conv2_Act_Desc;
	checkCUDNN(cudnnCreateActivationDescriptor(&conv2_Act_Desc));
	checkCUDNN(cudnnSetActivationDescriptor(conv2_Act_Desc, fwActalgo, CUDNN_PROPAGATE_NAN, 0));

	cudnnPoolingDescriptor_t pool2_Desc;
	checkCUDNN(cudnnCreatePoolingDescriptor(&pool2_Desc));
	checkCUDNN(cudnnSetPooling2dDescriptor(pool2_Desc, poolMode, CUDNN_PROPAGATE_NAN, poolWind_h, poolWind_w, poolPad_h, poolPad_w, poolStrd_h, poolStrd_w));

	cudnnTensorDescriptor_t pool2_Tensor;
	checkCUDNN(cudnnCreateTensorDescriptor(&pool2_Tensor));
	int pool2Out_n, pool2Out_c, pool2Out_h, pool2Out_w;
	checkCUDNN(cudnnGetPooling2dForwardOutputDim(pool2_Desc, conv2_Tensor, &pool2Out_n, &pool2Out_c, &pool2Out_h, &pool2Out_w));
	checkCUDNN(cudnnSetTensor4dDescriptor(pool2_Tensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, pool2Out_n, pool2Out_c, pool2Out_h, pool2Out_w));

	cudnnActivationDescriptor_t conv2pool_Act_Desc;
	checkCUDNN(cudnnCreateActivationDescriptor(&conv2pool_Act_Desc));
	checkCUDNN(cudnnSetActivationDescriptor(conv2pool_Act_Desc, fwActalgo, CUDNN_PROPAGATE_NAN, 0));

	//////////////3

	cudnnFilterDescriptor_t conv3_Filter_Desc;
	checkCUDNN(cudnnCreateFilterDescriptor(&conv3_Filter_Desc));
	checkCUDNN(cudnnSetFilter4dDescriptor(conv3_Filter_Desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, conv3FilterCnt, conv2FilterCnt, conv3FilterHeight, conv3FilterWidth));

	cudnnConvolutionDescriptor_t conv3_Desc;
	checkCUDNN(cudnnCreateConvolutionDescriptor(&conv3_Desc));
	checkCUDNN(cudnnSetConvolution2dDescriptor(conv3_Desc, conv3Pad_h, conv3Pad_w, conv3Str_h, conv3Str_w, conv3Dil_h, conv3Dil_w, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

	cudnnTensorDescriptor_t conv3_Tensor;
	checkCUDNN(cudnnCreateTensorDescriptor(&conv3_Tensor));
	int conv3Out_n, conv3Out_c, conv3Out_h, conv3Out_w;
	checkCUDNN(cudnnGetConvolution2dForwardOutputDim(conv3_Desc, pool2_Tensor, conv3_Filter_Desc, &conv3Out_n, &conv3Out_c, &conv3Out_h, &conv3Out_w));
	checkCUDNN(cudnnSetTensor4dDescriptor(conv3_Tensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, conv3Out_n, conv3Out_c, conv3Out_h, conv3Out_w));

	cudnnConvolutionFwdAlgo_t conv3_fwAlgo;
	checkCUDNN(cudnnGetConvolutionForwardAlgorithm(cudnn, pool2_Tensor, conv3_Filter_Desc, conv3_Desc, conv3_Tensor, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &conv3_fwAlgo));
	size_t conv3_worksSize = 0;
	checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn, pool2_Tensor, conv3_Filter_Desc, conv3_Desc, conv3_Tensor, conv3_fwAlgo, &conv3_worksSize));
	size_t * dev_conv3_works;
	checkCudaErrors(cudaMalloc((void**)&dev_conv3_works, conv3_worksSize));

	cudnnTensorDescriptor_t conv3_biasTensor;
	checkCUDNN(cudnnCreateTensorDescriptor(&conv3_biasTensor));
	checkCUDNN(cudnnSetTensor4dDescriptor(conv3_biasTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, conv3FilterCnt, 1, 1));

	cudnnActivationDescriptor_t conv3_Act_Desc;
	checkCUDNN(cudnnCreateActivationDescriptor(&conv3_Act_Desc));
	checkCUDNN(cudnnSetActivationDescriptor(conv3_Act_Desc, fwActalgo, CUDNN_PROPAGATE_NAN, 0));




	//////////////4

	cudnnFilterDescriptor_t conv4_Filter_Desc;
	checkCUDNN(cudnnCreateFilterDescriptor(&conv4_Filter_Desc));
	checkCUDNN(cudnnSetFilter4dDescriptor(conv4_Filter_Desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, conv4FilterCnt, conv3FilterCnt, conv4FilterHeight, conv4FilterWidth));

	cudnnConvolutionDescriptor_t conv4_Desc;
	checkCUDNN(cudnnCreateConvolutionDescriptor(&conv4_Desc));
	checkCUDNN(cudnnSetConvolution2dDescriptor(conv4_Desc, conv4Pad_h, conv4Pad_w, conv4Str_h, conv4Str_w, conv4Dil_h, conv4Dil_w, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

	cudnnTensorDescriptor_t conv4_Tensor;
	checkCUDNN(cudnnCreateTensorDescriptor(&conv4_Tensor));
	int conv4Out_n, conv4Out_c, conv4Out_h, conv4Out_w;
	checkCUDNN(cudnnGetConvolution2dForwardOutputDim(conv4_Desc, conv3_Tensor, conv4_Filter_Desc, &conv4Out_n, &conv4Out_c, &conv4Out_h, &conv4Out_w));
	checkCUDNN(cudnnSetTensor4dDescriptor(conv4_Tensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, conv4Out_n, conv4Out_c, conv4Out_h, conv4Out_w));

	cudnnConvolutionFwdAlgo_t conv4_fwAlgo;
	checkCUDNN(cudnnGetConvolutionForwardAlgorithm(cudnn, conv3_Tensor, conv4_Filter_Desc, conv4_Desc, conv4_Tensor, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &conv4_fwAlgo));
	size_t conv4_worksSize = 0;
	checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn, conv3_Tensor, conv4_Filter_Desc, conv4_Desc, conv4_Tensor, conv4_fwAlgo, &conv4_worksSize));
	size_t * dev_conv4_works;
	checkCudaErrors(cudaMalloc((void**)&dev_conv4_works, conv4_worksSize));

	cudnnTensorDescriptor_t conv4_biasTensor;
	checkCUDNN(cudnnCreateTensorDescriptor(&conv4_biasTensor));
	checkCUDNN(cudnnSetTensor4dDescriptor(conv4_biasTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, conv4FilterCnt, 1, 1));

	cudnnActivationDescriptor_t conv4_Act_Desc;
	checkCUDNN(cudnnCreateActivationDescriptor(&conv4_Act_Desc));
	checkCUDNN(cudnnSetActivationDescriptor(conv4_Act_Desc, fwActalgo, CUDNN_PROPAGATE_NAN, 0));




	//////////////5

	cudnnFilterDescriptor_t conv5_Filter_Desc;
	checkCUDNN(cudnnCreateFilterDescriptor(&conv5_Filter_Desc));
	checkCUDNN(cudnnSetFilter4dDescriptor(conv5_Filter_Desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, conv5FilterCnt, conv4FilterCnt, conv5FilterHeight, conv5FilterWidth));

	cudnnConvolutionDescriptor_t conv5_Desc;
	checkCUDNN(cudnnCreateConvolutionDescriptor(&conv5_Desc));
	checkCUDNN(cudnnSetConvolution2dDescriptor(conv5_Desc, conv5Pad_h, conv5Pad_w, conv5Str_h, conv5Str_w, conv5Dil_h, conv5Dil_w, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

	cudnnTensorDescriptor_t conv5_Tensor;
	checkCUDNN(cudnnCreateTensorDescriptor(&conv5_Tensor));
	int conv5Out_n, conv5Out_c, conv5Out_h, conv5Out_w;
	checkCUDNN(cudnnGetConvolution2dForwardOutputDim(conv5_Desc, conv4_Tensor, conv5_Filter_Desc, &conv5Out_n, &conv5Out_c, &conv5Out_h, &conv5Out_w));
	checkCUDNN(cudnnSetTensor4dDescriptor(conv5_Tensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, conv5Out_n, conv5Out_c, conv5Out_h, conv5Out_w));

	cudnnConvolutionFwdAlgo_t conv5_fwAlgo;
	checkCUDNN(cudnnGetConvolutionForwardAlgorithm(cudnn, conv4_Tensor, conv5_Filter_Desc, conv5_Desc, conv5_Tensor, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &conv5_fwAlgo));
	size_t conv5_worksSize = 0;
	checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn, conv4_Tensor, conv5_Filter_Desc, conv5_Desc, conv5_Tensor, conv5_fwAlgo, &conv5_worksSize));
	size_t * dev_conv5_works;
	checkCudaErrors(cudaMalloc((void**)&dev_conv5_works, conv5_worksSize));

	cudnnTensorDescriptor_t conv5_biasTensor;
	checkCUDNN(cudnnCreateTensorDescriptor(&conv5_biasTensor));
	checkCUDNN(cudnnSetTensor4dDescriptor(conv5_biasTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, conv5FilterCnt, 1, 1));

	cudnnActivationDescriptor_t conv5_Act_Desc;
	checkCUDNN(cudnnCreateActivationDescriptor(&conv5_Act_Desc));
	checkCUDNN(cudnnSetActivationDescriptor(conv5_Act_Desc, fwActalgo, CUDNN_PROPAGATE_NAN, 0));

	cudnnPoolingDescriptor_t pool5_Desc;
	checkCUDNN(cudnnCreatePoolingDescriptor(&pool5_Desc));
	checkCUDNN(cudnnSetPooling2dDescriptor(pool5_Desc, poolMode, CUDNN_PROPAGATE_NAN, poolWind_h, poolWind_w, poolPad_h, poolPad_w, poolStrd_h, poolStrd_w));

	cudnnTensorDescriptor_t pool5_Tensor;
	checkCUDNN(cudnnCreateTensorDescriptor(&pool5_Tensor));
	int pool5Out_n, pool5Out_c, pool5Out_h, pool5Out_w;
	checkCUDNN(cudnnGetPooling2dForwardOutputDim(pool5_Desc, conv5_Tensor, &pool5Out_n, &pool5Out_c, &pool5Out_h, &pool5Out_w));
	checkCUDNN(cudnnSetTensor4dDescriptor(pool5_Tensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, pool5Out_n, pool5Out_c, pool5Out_h, pool5Out_w));

	cudnnActivationDescriptor_t conv5pool_Act_Desc;
	checkCUDNN(cudnnCreateActivationDescriptor(&conv5pool_Act_Desc));
	checkCUDNN(cudnnSetActivationDescriptor(conv5pool_Act_Desc, fwActalgo, CUDNN_PROPAGATE_NAN, 0));

	////////////////////////

	cudnnFilterDescriptor_t fc1_Filter_Desc;
	checkCUDNN(cudnnCreateFilterDescriptor(&fc1_Filter_Desc));
	checkCUDNN(cudnnSetFilter4dDescriptor(fc1_Filter_Desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, fc1FilterCnt, conv5FilterCnt, pool5OutHeight, pool5OutWidth));

	cudnnConvolutionDescriptor_t fc1_Desc;
	checkCUDNN(cudnnCreateConvolutionDescriptor(&fc1_Desc));
	checkCUDNN(cudnnSetConvolution2dDescriptor(fc1_Desc, fcPad_h, fcPad_w, fcStr_h, fcStr_w, fcDil_h, fcDil_w, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

	cudnnTensorDescriptor_t fc1_Tensor;
	checkCUDNN(cudnnCreateTensorDescriptor(&fc1_Tensor));
	int fwd1Out_n, fwd1Out_c, fwd1Out_h, fwd1Out_w;
	checkCUDNN(cudnnGetConvolution2dForwardOutputDim(fc1_Desc, pool5_Tensor, fc1_Filter_Desc, &fwd1Out_n, &fwd1Out_c, &fwd1Out_h, &fwd1Out_w));
	checkCUDNN(cudnnSetTensor4dDescriptor(fc1_Tensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, fwd1Out_n, fwd1Out_c, fwd1Out_h, fwd1Out_w));

	cudnnConvolutionFwdAlgo_t fc1_fwAlgo;
	checkCUDNN(cudnnGetConvolutionForwardAlgorithm(cudnn, pool5_Tensor, fc1_Filter_Desc, fc1_Desc, fc1_Tensor, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &fc1_fwAlgo));
	size_t fc1_worksSize = 0;
	checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn, pool5_Tensor, fc1_Filter_Desc, fc1_Desc, fc1_Tensor, fc1_fwAlgo, &fc1_worksSize));
	size_t * dev_fc1_works;
	checkCudaErrors(cudaMalloc((void**)&dev_fc1_works, fc1_worksSize));

	cudnnTensorDescriptor_t fc1_biasTensor;
	checkCUDNN(cudnnCreateTensorDescriptor(&fc1_biasTensor));
	checkCUDNN(cudnnSetTensor4dDescriptor(fc1_biasTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, fc1FilterCnt, 1, 1));

	cudnnActivationDescriptor_t fc1_Act_Desc;
	checkCUDNN(cudnnCreateActivationDescriptor(&fc1_Act_Desc));
	checkCUDNN(cudnnSetActivationDescriptor(fc1_Act_Desc, fwActalgo, CUDNN_PROPAGATE_NAN, 0));

	cudnnFilterDescriptor_t fc2_Filter_Desc;
	checkCUDNN(cudnnCreateFilterDescriptor(&fc2_Filter_Desc));
	checkCUDNN(cudnnSetFilter4dDescriptor(fc2_Filter_Desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, fc2FilterCnt, fc1FilterCnt, 1, 1));

	cudnnConvolutionDescriptor_t fc2_Desc;
	checkCUDNN(cudnnCreateConvolutionDescriptor(&fc2_Desc));
	checkCUDNN(cudnnSetConvolution2dDescriptor(fc2_Desc, fcPad_h, fcPad_w, fcStr_h, fcStr_w, fcDil_h, fcDil_w, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

	cudnnTensorDescriptor_t fc2_Tensor;
	checkCUDNN(cudnnCreateTensorDescriptor(&fc2_Tensor));
	int fwd2Out_n, fwd2Out_c, fwd2Out_h, fwd2Out_w;
	checkCUDNN(cudnnGetConvolution2dForwardOutputDim(fc2_Desc, fc1_Tensor, fc2_Filter_Desc, &fwd2Out_n, &fwd2Out_c, &fwd2Out_h, &fwd2Out_w));
	checkCUDNN(cudnnSetTensor4dDescriptor(fc2_Tensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, fwd2Out_n, fwd2Out_c, fwd2Out_h, fwd2Out_w));

	cudnnConvolutionFwdAlgo_t fc2_fwAlgo;
	checkCUDNN(cudnnGetConvolutionForwardAlgorithm(cudnn, fc1_Tensor, fc2_Filter_Desc, fc2_Desc, fc2_Tensor, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &fc2_fwAlgo));
	size_t fc2_worksSize = 0;
	checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn, fc1_Tensor, fc2_Filter_Desc, fc2_Desc, fc2_Tensor, fc2_fwAlgo, &fc2_worksSize));
	size_t * dev_fc2_works;
	checkCudaErrors(cudaMalloc((void**)&dev_fc2_works, fc2_worksSize));

	cudnnTensorDescriptor_t fc2_biasTensor;
	checkCUDNN(cudnnCreateTensorDescriptor(&fc2_biasTensor));
	checkCUDNN(cudnnSetTensor4dDescriptor(fc2_biasTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, fc2FilterCnt, 1, 1));

	cudnnActivationDescriptor_t fc2_Act_Desc;
	checkCUDNN(cudnnCreateActivationDescriptor(&fc2_Act_Desc));
	checkCUDNN(cudnnSetActivationDescriptor(fc2_Act_Desc, fwActalgo, CUDNN_PROPAGATE_NAN, 0));

	cudnnFilterDescriptor_t fc3_Filter_Desc;
	checkCUDNN(cudnnCreateFilterDescriptor(&fc3_Filter_Desc));
	checkCUDNN(cudnnSetFilter4dDescriptor(fc3_Filter_Desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, lastLayer_numOut, fc2FilterCnt, 1, 1));

	cudnnConvolutionDescriptor_t fc3_Desc;
	checkCUDNN(cudnnCreateConvolutionDescriptor(&fc3_Desc));
	checkCUDNN(cudnnSetConvolution2dDescriptor(fc3_Desc, fcPad_h, fcPad_w, fcStr_h, fcStr_w, fcDil_h, fcDil_w, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

	cudnnTensorDescriptor_t fc3_Tensor;
	checkCUDNN(cudnnCreateTensorDescriptor(&fc3_Tensor));
	int fwd3Out_n, fwd3Out_c, fwd3Out_h, fwd3Out_w;
	checkCUDNN(cudnnGetConvolution2dForwardOutputDim(fc3_Desc, fc2_Tensor, fc3_Filter_Desc, &fwd3Out_n, &fwd3Out_c, &fwd3Out_h, &fwd3Out_w));
	checkCUDNN(cudnnSetTensor4dDescriptor(fc3_Tensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, fwd3Out_n, fwd3Out_c, fwd3Out_h, fwd3Out_w));

	cudnnConvolutionFwdAlgo_t fc3_fwAlgo;
	checkCUDNN(cudnnGetConvolutionForwardAlgorithm(cudnn, fc2_Tensor, fc3_Filter_Desc, fc3_Desc, fc3_Tensor, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &fc3_fwAlgo));
	size_t fc3_worksSize = 0;
	checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn, fc2_Tensor, fc3_Filter_Desc, fc3_Desc, fc3_Tensor, fc3_fwAlgo, &fc3_worksSize));
	size_t * dev_fc3_works;
	checkCudaErrors(cudaMalloc((void**)&dev_fc3_works, fc3_worksSize));

	cudnnTensorDescriptor_t fc3_biasTensor;
	checkCUDNN(cudnnCreateTensorDescriptor(&fc3_biasTensor));
	checkCUDNN(cudnnSetTensor4dDescriptor(fc3_biasTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, lastLayer_numOut, 1, 1));


	/**********BACKPWARD**********/

	float* dev_bw_fc3Bias;
	checkCudaErrors(cudaMalloc((void**)&dev_bw_fc3Bias, sizeof(float) * 1 * lastLayer_numOut * 1 * 1));

	float* dev_bwf_fc3_Output;
	checkCudaErrors(cudaMalloc((void**)&dev_bwf_fc3_Output, sizeof(float) * lastLayer_numOut * fc2FilterCnt * 1 * 1));

	float* dev_bwd_fc3_Output;
	checkCudaErrors(cudaMalloc((void**)&dev_bwd_fc3_Output, sizeof(float) * batchSize * fc2FilterCnt * 1 * 1));

	float* dev_bw_fc2_ActDelta;
	checkCudaErrors(cudaMalloc((void**)&dev_bw_fc2_ActDelta, sizeof(float) * batchSize * fc2FilterCnt * 1 * 1));

	float* dev_bw_fc2Bias;
	checkCudaErrors(cudaMalloc((void**)&dev_bw_fc2Bias, sizeof(float) * 1 * fc2FilterCnt * 1 * 1));

	float* dev_bwf_fc2_Output;
	checkCudaErrors(cudaMalloc((void**)&dev_bwf_fc2_Output, sizeof(float) * fc2FilterCnt * fc1FilterCnt * 1 * 1));

	float* dev_bwd_fc2_Output;
	checkCudaErrors(cudaMalloc((void**)&dev_bwd_fc2_Output, sizeof(float) * batchSize * fc1FilterCnt * 1 * 1));

	float* dev_bw_fc1_ActDelta;
	checkCudaErrors(cudaMalloc((void**)&dev_bw_fc1_ActDelta, sizeof(float) * batchSize * fc1FilterCnt * 1 * 1));

	float* dev_bw_fc1Bias;
	checkCudaErrors(cudaMalloc((void**)&dev_bw_fc1Bias, sizeof(float) * 1 * fc1FilterCnt * 1 * 1));

	float* dev_bwf_fc1_Output;
	checkCudaErrors(cudaMalloc((void**)&dev_bwf_fc1_Output, sizeof(float) * fc1FilterCnt * conv5FilterCnt * pool5OutHeight * pool5OutWidth));

	float* dev_bwd_fc1_Output;
	checkCudaErrors(cudaMalloc((void**)&dev_bwd_fc1_Output, sizeof(float) * batchSize *  conv5FilterCnt * pool5OutHeight * pool5OutWidth));

	// conv 5 back

	float* dev_bw_conv5pool_ActDelta;
	checkCudaErrors(cudaMalloc((void**)&dev_bw_conv5pool_ActDelta, sizeof(float) * batchSize * conv5FilterCnt * pool5OutHeight * pool5OutWidth));

	float* dev_pool5_Delta;
	checkCudaErrors(cudaMalloc((void**)&dev_pool5_Delta, sizeof(float) * batchSize * conv5FilterCnt * conv5OutHeight * conv5OutWidth));

	float* dev_bw_conv5_ActDelta;
	checkCudaErrors(cudaMalloc((void**)&dev_bw_conv5_ActDelta, sizeof(float) * batchSize * conv5FilterCnt * conv5OutHeight * conv5OutWidth));

	float* dev_bw_conv5Bias;
	checkCudaErrors(cudaMalloc((void**)&dev_bw_conv5Bias, sizeof(float) * 1 * conv5FilterCnt * 1 * 1));

	float* dev_bwf_conv5_Output;
	checkCudaErrors(cudaMalloc((void**)&dev_bwf_conv5_Output, sizeof(float) * conv5FilterCnt * conv4FilterCnt * conv5FilterHeight * conv5FilterWidth));

	float* dev_bwd_conv5_Output;
	checkCudaErrors(cudaMalloc((void**)&dev_bwd_conv5_Output, sizeof(float) * batchSize * conv4FilterCnt * conv4OutHeight * conv4OutWidth));


	// conv 4 back


	float* dev_bw_conv4_ActDelta;
	checkCudaErrors(cudaMalloc((void**)&dev_bw_conv4_ActDelta, sizeof(float) * batchSize * conv4FilterCnt * conv4OutHeight * conv4OutWidth));

	float* dev_bw_conv4Bias;
	checkCudaErrors(cudaMalloc((void**)&dev_bw_conv4Bias, sizeof(float) * 1 * conv4FilterCnt * 1 * 1));

	float* dev_bwf_conv4_Output;
	checkCudaErrors(cudaMalloc((void**)&dev_bwf_conv4_Output, sizeof(float) * conv4FilterCnt * conv3FilterCnt * conv4FilterHeight * conv4FilterWidth));

	float* dev_bwd_conv4_Output;
	checkCudaErrors(cudaMalloc((void**)&dev_bwd_conv4_Output, sizeof(float) * batchSize * conv3FilterCnt * conv3OutHeight * conv3OutWidth));



	// conv 3 back

	float* dev_bw_conv3_ActDelta;
	checkCudaErrors(cudaMalloc((void**)&dev_bw_conv3_ActDelta, sizeof(float) * batchSize * conv3FilterCnt * conv3OutHeight * conv3OutWidth));

	float* dev_bw_conv3Bias;
	checkCudaErrors(cudaMalloc((void**)&dev_bw_conv3Bias, sizeof(float) * 1 * conv3FilterCnt * 1 * 1));

	float* dev_bwf_conv3_Output;
	checkCudaErrors(cudaMalloc((void**)&dev_bwf_conv3_Output, sizeof(float) * conv3FilterCnt * conv2FilterCnt * conv3FilterHeight * conv3FilterWidth));

	float* dev_bwd_conv3_Output;
	checkCudaErrors(cudaMalloc((void**)&dev_bwd_conv3_Output, sizeof(float) * batchSize * conv2FilterCnt * pool2OutHeight * pool2OutWidth));

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// conv 2 back

	float* dev_bw_conv2pool_ActDelta;
	checkCudaErrors(cudaMalloc((void**)&dev_bw_conv2pool_ActDelta, sizeof(float) * batchSize * conv2FilterCnt * pool2OutHeight * pool2OutWidth));

	float* dev_pool2_Delta;
	checkCudaErrors(cudaMalloc((void**)&dev_pool2_Delta, sizeof(float) * batchSize * conv2FilterCnt * conv2OutHeight * conv2OutWidth));

	float* dev_bw_conv2_ActDelta;
	checkCudaErrors(cudaMalloc((void**)&dev_bw_conv2_ActDelta, sizeof(float) * batchSize * conv2FilterCnt * conv2OutHeight * conv2OutWidth));

	float* dev_bw_conv2Bias;
	checkCudaErrors(cudaMalloc((void**)&dev_bw_conv2Bias, sizeof(float) * 1 * conv2FilterCnt * 1 * 1));

	float* dev_bwf_conv2_Output;
	checkCudaErrors(cudaMalloc((void**)&dev_bwf_conv2_Output, sizeof(float) * conv2FilterCnt * conv1FilterCnt * conv2FilterHeight * conv2FilterWidth));

	float* dev_bwd_conv2_Output;
	checkCudaErrors(cudaMalloc((void**)&dev_bwd_conv2_Output, sizeof(float) * batchSize * conv1FilterCnt * pool1OutHeight * pool1OutWidth));




	float* dev_bw_conv1pool_ActDelta;
	checkCudaErrors(cudaMalloc((void**)&dev_bw_conv1pool_ActDelta, sizeof(float) * batchSize * conv1FilterCnt * pool1OutHeight * pool1OutWidth));

	float* dev_pool1_Delta;
	checkCudaErrors(cudaMalloc((void**)&dev_pool1_Delta, sizeof(float) * batchSize * conv1FilterCnt * conv1OutHeight * conv1OutWidth));

	float* dev_bw_conv1_ActDelta;
	checkCudaErrors(cudaMalloc((void**)&dev_bw_conv1_ActDelta, sizeof(float) * batchSize * conv1FilterCnt * conv1OutHeight * conv1OutWidth));

	float* dev_bw_conv1Bias;
	checkCudaErrors(cudaMalloc((void**)&dev_bw_conv1Bias, sizeof(float) * 1 * conv1FilterCnt * 1 * 1));

	float* dev_bwf_conv1_Output;
	checkCudaErrors(cudaMalloc((void**)&dev_bwf_conv1_Output, sizeof(float) * conv1FilterCnt * input_channelCnt * conv1FilterHeight * conv1FilterWidth));


	cudnnConvolutionBwdFilterAlgo_t fc3_bwFAlgo;
	checkCUDNN(cudnnGetConvolutionBackwardFilterAlgorithm(cudnn, fc2_Tensor, fc3_Tensor, fc3_Desc, fc3_Filter_Desc, CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, &fc3_bwFAlgo));
	size_t fc3_bwFworksSize = 0;
	checkCUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnn, fc2_Tensor, fc3_Tensor, fc3_Desc, fc3_Filter_Desc, fc3_bwFAlgo, &fc3_bwFworksSize));
	size_t* dev_fc3_bwFworks;
	checkCudaErrors(cudaMalloc((void**)&dev_fc3_bwFworks, fc3_bwFworksSize));

	cudnnConvolutionBwdDataAlgo_t fc3_bwDAlgo;
	checkCUDNN(cudnnGetConvolutionBackwardDataAlgorithm(cudnn, fc3_Filter_Desc, fc3_Tensor, fc3_Desc, fc2_Tensor, CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, &fc3_bwDAlgo));
	size_t fc3_bwDworksSize = 0;
	checkCUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn, fc3_Filter_Desc, fc3_Tensor, fc3_Desc, fc2_Tensor, fc3_bwDAlgo, &fc3_bwDworksSize));
	size_t* dev_fc3_bwDworks;
	checkCudaErrors(cudaMalloc((void**)&dev_fc3_bwDworks, fc3_bwDworksSize));

	cudnnConvolutionBwdFilterAlgo_t fc2_bwFAlgo;
	checkCUDNN(cudnnGetConvolutionBackwardFilterAlgorithm(cudnn, fc1_Tensor, fc2_Tensor, fc2_Desc, fc2_Filter_Desc, CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, &fc2_bwFAlgo));
	size_t fc2_bwFworksSize = 0;
	checkCUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnn, fc1_Tensor, fc2_Tensor, fc2_Desc, fc2_Filter_Desc, fc2_bwFAlgo, &fc2_bwFworksSize));
	size_t* dev_fc2_bwFworks;
	checkCudaErrors(cudaMalloc((void**)&dev_fc2_bwFworks, fc2_bwFworksSize));

	cudnnConvolutionBwdDataAlgo_t fc2_bwDAlgo;
	checkCUDNN(cudnnGetConvolutionBackwardDataAlgorithm(cudnn, fc2_Filter_Desc, fc2_Tensor, fc2_Desc, fc1_Tensor, CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, &fc2_bwDAlgo));
	size_t fc2_bwDworksSize = 0;
	checkCUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn, fc2_Filter_Desc, fc2_Tensor, fc2_Desc, fc1_Tensor, fc2_bwDAlgo, &fc2_bwDworksSize));
	size_t* dev_fc2_bwDworks;
	checkCudaErrors(cudaMalloc((void**)&dev_fc2_bwDworks, fc2_bwDworksSize));

	cudnnConvolutionBwdFilterAlgo_t fc1_bwFAlgo;
	checkCUDNN(cudnnGetConvolutionBackwardFilterAlgorithm(cudnn, conv5_Tensor, fc1_Tensor, fc1_Desc, fc1_Filter_Desc, CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, &fc1_bwFAlgo));
	size_t fc1_bwFworksSize = 0;
	checkCUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnn, conv5_Tensor, fc1_Tensor, fc1_Desc, fc1_Filter_Desc, fc1_bwFAlgo, &fc1_bwFworksSize));
	size_t* dev_fc1_bwFworks;
	checkCudaErrors(cudaMalloc((void**)&dev_fc1_bwFworks, fc1_bwFworksSize));

	cudnnConvolutionBwdDataAlgo_t fc1_bwDAlgo;
	checkCUDNN(cudnnGetConvolutionBackwardDataAlgorithm(cudnn, fc1_Filter_Desc, fc1_Tensor, fc1_Desc, conv5_Tensor, CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, &fc1_bwDAlgo));
	size_t fc1_bwDworksSize = 0;
	checkCUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn, fc1_Filter_Desc, fc1_Tensor, fc1_Desc, conv5_Tensor, fc1_bwDAlgo, &fc1_bwDworksSize));
	size_t* dev_fc1_bwDworks;
	checkCudaErrors(cudaMalloc((void**)&dev_fc1_bwDworks, fc1_bwDworksSize));

	//5
	cudnnConvolutionBwdFilterAlgo_t conv5_bwFAlgo;
	checkCUDNN(cudnnGetConvolutionBackwardFilterAlgorithm(cudnn, conv4_Tensor, conv5_Tensor, conv5_Desc, conv5_Filter_Desc, CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, &conv5_bwFAlgo));
	size_t conv5_bwFworksSize = 0;
	checkCUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnn, conv4_Tensor, conv5_Tensor, conv5_Desc, conv5_Filter_Desc, conv5_bwFAlgo, &conv5_bwFworksSize));
	size_t * dev_conv5_bwFworks;
	checkCudaErrors(cudaMalloc((void**)&dev_conv5_bwFworks, conv5_bwFworksSize));

	cudnnConvolutionBwdDataAlgo_t conv5_bwDAlgo;
	checkCUDNN(cudnnGetConvolutionBackwardDataAlgorithm(cudnn, conv5_Filter_Desc, conv5_Tensor, conv5_Desc, conv4_Tensor, CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, &conv5_bwDAlgo));
	size_t conv5_bwDworksSize = 0;
	checkCUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn, conv5_Filter_Desc, conv5_Tensor, conv5_Desc, conv4_Tensor, conv5_bwDAlgo, &conv5_bwDworksSize));
	size_t* dev_conv5_bwDworks;
	checkCudaErrors(cudaMalloc((void**)&dev_conv5_bwDworks, conv5_bwDworksSize));

	//4

	cudnnConvolutionBwdFilterAlgo_t conv4_bwFAlgo;
	checkCUDNN(cudnnGetConvolutionBackwardFilterAlgorithm(cudnn, conv3_Tensor, conv4_Tensor, conv4_Desc, conv4_Filter_Desc, CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, &conv4_bwFAlgo));
	size_t conv4_bwFworksSize = 0;
	checkCUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnn, conv3_Tensor, conv4_Tensor, conv4_Desc, conv4_Filter_Desc, conv4_bwFAlgo, &conv4_bwFworksSize));
	size_t * dev_conv4_bwFworks;
	checkCudaErrors(cudaMalloc((void**)&dev_conv4_bwFworks, conv4_bwFworksSize));

	cudnnConvolutionBwdDataAlgo_t conv4_bwDAlgo;
	checkCUDNN(cudnnGetConvolutionBackwardDataAlgorithm(cudnn, conv4_Filter_Desc, conv4_Tensor, conv4_Desc, conv3_Tensor, CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, &conv4_bwDAlgo));
	size_t conv4_bwDworksSize = 0;
	checkCUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn, conv4_Filter_Desc, conv4_Tensor, conv4_Desc, conv3_Tensor, conv4_bwDAlgo, &conv4_bwDworksSize));
	size_t* dev_conv4_bwDworks;
	checkCudaErrors(cudaMalloc((void**)&dev_conv4_bwDworks, conv4_bwDworksSize));


	//3

	cudnnConvolutionBwdFilterAlgo_t conv3_bwFAlgo;
	checkCUDNN(cudnnGetConvolutionBackwardFilterAlgorithm(cudnn, pool2_Tensor, conv3_Tensor, conv3_Desc, conv3_Filter_Desc, CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, &conv3_bwFAlgo));
	size_t conv3_bwFworksSize = 0;
	checkCUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnn, pool2_Tensor, conv3_Tensor, conv3_Desc, conv3_Filter_Desc, conv3_bwFAlgo, &conv3_bwFworksSize));
	size_t * dev_conv3_bwFworks;
	checkCudaErrors(cudaMalloc((void**)&dev_conv3_bwFworks, conv3_bwFworksSize));

	cudnnConvolutionBwdDataAlgo_t conv3_bwDAlgo;
	checkCUDNN(cudnnGetConvolutionBackwardDataAlgorithm(cudnn, conv3_Filter_Desc, conv3_Tensor, conv3_Desc, pool2_Tensor, CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, &conv3_bwDAlgo));
	size_t conv3_bwDworksSize = 0;
	checkCUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn, conv3_Filter_Desc, conv3_Tensor, conv3_Desc, pool2_Tensor, conv3_bwDAlgo, &conv3_bwDworksSize));
	size_t* dev_conv3_bwDworks;
	checkCudaErrors(cudaMalloc((void**)&dev_conv3_bwDworks, conv3_bwDworksSize));

	//2


	cudnnConvolutionBwdFilterAlgo_t conv2_bwFAlgo;
	checkCUDNN(cudnnGetConvolutionBackwardFilterAlgorithm(cudnn, pool1_Tensor, conv2_Tensor, conv2_Desc, conv2_Filter_Desc, CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, &conv2_bwFAlgo));
	size_t conv2_bwFworksSize = 0;
	checkCUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnn, pool1_Tensor, conv2_Tensor, conv2_Desc, conv2_Filter_Desc, conv2_bwFAlgo, &conv2_bwFworksSize));
	size_t * dev_conv2_bwFworks;
	checkCudaErrors(cudaMalloc((void**)&dev_conv2_bwFworks, conv2_bwFworksSize));

	cudnnConvolutionBwdDataAlgo_t conv2_bwDAlgo;
	checkCUDNN(cudnnGetConvolutionBackwardDataAlgorithm(cudnn, conv2_Filter_Desc, conv2_Tensor, conv2_Desc, pool1_Tensor, CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, &conv2_bwDAlgo));
	size_t conv2_bwDworksSize = 0;
	checkCUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn, conv2_Filter_Desc, conv2_Tensor, conv2_Desc, pool1_Tensor, conv2_bwDAlgo, &conv2_bwDworksSize));
	size_t* dev_conv2_bwDworks;
	checkCudaErrors(cudaMalloc((void**)&dev_conv2_bwDworks, conv2_bwDworksSize));


	//1


	cudnnConvolutionBwdFilterAlgo_t conv1_bwFAlgo;
	checkCUDNN(cudnnGetConvolutionBackwardFilterAlgorithm(cudnn, input_Tensor, conv1_Tensor, conv1_Desc, conv1_Filter_Desc, CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, &conv1_bwFAlgo));
	size_t conv1_bwFworksSize = 0;
	checkCUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnn, input_Tensor, conv1_Tensor, conv1_Desc, conv1_Filter_Desc, conv1_bwFAlgo, &conv1_bwFworksSize));
	size_t * dev_conv1_bwFworks;
	checkCudaErrors(cudaMalloc((void**)&dev_conv1_bwFworks, conv1_bwFworksSize));



	//weight update

	//float momentum = 0.9;
	//float eta = -1.0;
	float * dev_dloss;


	float* dev_target;
	cudaMalloc((void**)&dev_target, sizeof(float) * batchSize);

	float* dev_input;
	cudaMalloc((void**)&dev_input, sizeof(float) * batchSize * input_channelCnt * imageHeight * imageWidth);

	cout << "======================  학습 시작  ==========================" << endl;

	startTime_train = clock();

	for (int iter = 0; iter < epoch; iter++)
	{


		float* yhat = new float[batchSize* num_labels];// soft max 결과 값 
		int* predicted = new int[ImageNum]; // 소프트 맥스 결과 값에서 가장 큰 값이 들어 있는 위치(인덱스)를 저장하는 배열
		int count = 0;


		for (int a = 0; a < ImageNum / batchSize; a++) { // 배치 계산 루프 

			for (int i = 0; i < batchSize; i++) {
				unsigned char* temp_train = ImgBox[i + (batchSize * a)].first.data;
				for (int c = 0; c < input_channelCnt; c++) {
					for (int y = 0; y < imageHeight; y++) {
						for (int x = 0; x < imageWidth; x++) {
							Input_train[i * input_channelCnt * imageHeight * imageWidth + c * imageHeight * imageWidth + y * imageWidth + x] = temp_train[input_channelCnt * imageHeight * x + input_channelCnt * y + c] / 255.0;
						}
					}
				}

			}

			for (int i = 0; i < batchSize; i++)
			{
				target_train_batch[i] = target_train[i + (batchSize * a)];
			}



			cudaMemcpy(dev_target, target_train_batch, sizeof(float) * batchSize, cudaMemcpyHostToDevice);
			cudaMemcpy(dev_input, Input_train, sizeof(float) * batchSize * input_channelCnt * imageHeight * imageWidth, cudaMemcpyHostToDevice);

			//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			//GPU train 계산 들어갈 자리 
			cudnnConvolutionForward(cudnn, &alpha, input_Tensor, dev_input, conv1_Filter_Desc, dev_conv1_Filter, conv1_Desc, conv1_fwAlgo, dev_conv1_works, conv1_worksSize, &beta, conv1_Tensor, dev_conv1_Output);
			cudnnAddTensor(cudnn, &alpha, conv1_biasTensor, dev_conv1_Bias, &alpha, conv1_Tensor, dev_conv1_Output);
			cudnnActivationForward(cudnn, conv1_Act_Desc, &alpha, conv1_Tensor, dev_conv1_Output, &beta, conv1_Tensor, dev_conv1Act_output);
			cudnnPoolingForward(cudnn, pool1_Desc, &alpha, conv1_Tensor, dev_conv1Act_output, &beta, pool1_Tensor, dev_pool1_Output);
			cudnnActivationForward(cudnn, conv1pool_Act_Desc, &alpha, pool1_Tensor, dev_pool1_Output, &beta, pool1_Tensor, dev_conv1pool_Act_output);
			
			
			//conv2 forward
			cudnnConvolutionForward(cudnn, &alpha, pool1_Tensor, dev_conv1pool_Act_output, conv2_Filter_Desc, dev_conv2_Filter, conv2_Desc, conv2_fwAlgo, dev_conv2_works, conv2_worksSize, &beta, conv2_Tensor, dev_conv2_Output);
			cudnnAddTensor(cudnn, &alpha, conv2_biasTensor, dev_conv2_Bias, &alpha, conv2_Tensor, dev_conv2_Output);
			cudnnActivationForward(cudnn, conv2_Act_Desc, &alpha, conv2_Tensor, dev_conv2_Output, &beta, conv2_Tensor, dev_conv2Act_output);
			cudnnPoolingForward(cudnn, pool2_Desc, &alpha, conv2_Tensor, dev_conv2Act_output, &beta, pool2_Tensor, dev_pool2_Output);
			cudnnActivationForward(cudnn, conv2pool_Act_Desc, &alpha, pool2_Tensor, dev_pool2_Output, &beta, pool2_Tensor, dev_conv2pool_Act_output);


			//conv3 forward
			cudnnConvolutionForward(cudnn, &alpha, pool2_Tensor, dev_conv2pool_Act_output, conv3_Filter_Desc, dev_conv3_Filter, conv3_Desc, conv3_fwAlgo, dev_conv3_works, conv3_worksSize, &beta, conv3_Tensor, dev_conv3_Output);
			cudnnAddTensor(cudnn, &alpha, conv3_biasTensor, dev_conv3_Bias, &alpha, conv3_Tensor, dev_conv3_Output);
			cudnnActivationForward(cudnn, conv3_Act_Desc, &alpha, conv3_Tensor, dev_conv3_Output, &beta, conv3_Tensor, dev_conv3Act_output);


			//conv4 forward
			cudnnConvolutionForward(cudnn, &alpha, conv3_Tensor, dev_conv3Act_output, conv4_Filter_Desc, dev_conv4_Filter, conv4_Desc, conv4_fwAlgo, dev_conv4_works, conv4_worksSize, &beta, conv4_Tensor, dev_conv4_Output);
			cudnnAddTensor(cudnn, &alpha, conv4_biasTensor, dev_conv4_Bias, &alpha, conv4_Tensor, dev_conv4_Output);
			cudnnActivationForward(cudnn, conv4_Act_Desc, &alpha, conv4_Tensor, dev_conv4_Output, &beta, conv4_Tensor, dev_conv4Act_output);


			//conv5 forward
			cudnnConvolutionForward(cudnn, &alpha, conv4_Tensor, dev_conv4Act_output, conv5_Filter_Desc, dev_conv5_Filter, conv5_Desc, conv5_fwAlgo, dev_conv5_works, conv5_worksSize, &beta, conv5_Tensor, dev_conv5_Output);
			cudnnAddTensor(cudnn, &alpha, conv5_biasTensor, dev_conv5_Bias, &alpha, conv5_Tensor, dev_conv5_Output);
			cudnnActivationForward(cudnn, conv5_Act_Desc, &alpha, conv5_Tensor, dev_conv5_Output, &beta, conv5_Tensor, dev_conv5Act_output);
			cudnnPoolingForward(cudnn, pool5_Desc, &alpha, conv5_Tensor, dev_conv5Act_output, &beta, pool5_Tensor, dev_pool5_Output);
			cudnnActivationForward(cudnn, conv5pool_Act_Desc, &alpha, pool5_Tensor, dev_pool5_Output, &beta, pool5_Tensor, dev_conv5pool_Act_output);



			//fc1 forward
			cudnnConvolutionForward(cudnn, &alpha, pool5_Tensor, dev_conv5pool_Act_output, fc1_Filter_Desc, dev_fc1_Filter, fc1_Desc, fc1_fwAlgo, dev_fc1_works, fc1_worksSize, &beta, fc1_Tensor, dev_fc1_Output);
			cudnnAddTensor(cudnn, &alpha, fc1_biasTensor, dev_fc1_Bias, &alpha, fc1_Tensor, dev_fc1_Output);
			cudnnActivationForward(cudnn, fc1_Act_Desc, &alpha, fc1_Tensor, dev_fc1_Output, &beta, fc1_Tensor, dev_fc1_Actout);


			//fc2 forward
			cudnnConvolutionForward(cudnn, &alpha, fc1_Tensor, dev_fc1_Actout, fc2_Filter_Desc, dev_fc2_Filter, fc2_Desc, fc2_fwAlgo, dev_fc2_works, fc2_worksSize, &beta, fc2_Tensor, dev_fc2_Output);
			cudnnAddTensor(cudnn, &alpha, fc2_biasTensor, dev_fc2_Bias, &alpha, fc2_Tensor, dev_fc2_Output);
			cudnnActivationForward(cudnn, fc2_Act_Desc, &alpha, fc2_Tensor, dev_fc2_Output, &beta, fc2_Tensor, dev_fc2_Actout);


			//fc3 forward
			cudnnConvolutionForward(cudnn, &alpha, fc2_Tensor, dev_fc2_Actout, fc3_Filter_Desc, dev_fc3_Filter, fc3_Desc, fc3_fwAlgo, dev_fc3_works, fc3_worksSize, &beta, fc3_Tensor, dev_fc3_Output);
			cudnnAddTensor(cudnn, &alpha, fc3_biasTensor, dev_fc3_Bias, &alpha, fc3_Tensor, dev_fc3_Output);
			cudnnSoftmaxForward(cudnn, sftAlgo, sftMode, &alpha, fc3_Tensor, dev_fc3_Output, &beta, fc3_Tensor, dev_smaxOutput);
			dev_dloss = dev_smaxOutput;





			cudaMemcpy(yhat, dev_smaxOutput, sizeof(float) * batchSize * num_labels, cudaMemcpyDeviceToHost);

			//one hot 위치 찾기


			for (size_t i = 0; i < batchSize; i++) {
				float temp = yhat[i * 10];// 예측된 첫번째 값을 임시 변수에 저장
				int indexJ = 0; // 가장 큰값을 찾기 위해 사용 되는 위치 인덱스

				for (size_t j = 0; j < num_labels - 1; j++) {
					if (temp > yhat[i * 10 + j + 1]) // 임시 변수에 넣어준 값과 비교
					{
						yhat[i * 10 + j + 1] = 0; // 임시 변수에 들어 있는 값보다 작다면 0 입력
					}
					else                      // 임시 변수에 들어 있는 값보다 크다면
					{
						temp = yhat[i * 10 + j + 1]; // 임시 변수에 해당 값을 저장
						yhat[i * 10 + indexJ] = 0; // 임시 변수에 이전에 들어 있던 값의 인덱스를 이용하여 이전 값 위치에 0 입력
						indexJ = j + 1; // 제일 큰 값(정답 예측 값) 위치를 인덱스 변수에 저장
					}
				}

				predicted[i] = indexJ; // 해당 이미지의 소프트 맥스 값중 가장 큰 값을 갖는 위치 인덱스를 배열에 저장
			}








			// 커널 함수 ( 오차(=dloss=dy)값 계산) 
			SoftmaxLossBackprop << <(batchSize + BW - 1) / BW, BW >> > (dev_target, num_labels, batchSize, dev_dloss);


			//fc3 back
			cudnnConvolutionBackwardBias(cudnn, &alpha, fc3_Tensor, dev_dloss, &beta, fc3_biasTensor, dev_bw_fc3Bias);
			cudnnConvolutionBackwardFilter(cudnn, &alpha, fc2_Tensor, dev_fc2_Actout, fc3_Tensor, dev_dloss, fc3_Desc,
				fc3_bwFAlgo, dev_fc3_bwFworks, fc3_bwFworksSize, &beta, fc3_Filter_Desc, dev_bwf_fc3_Output);
			cudnnConvolutionBackwardData(cudnn, &alpha, fc3_Filter_Desc, dev_fc3_Filter, fc3_Tensor, dev_dloss,
				fc3_Desc, fc3_bwDAlgo, dev_fc3_bwDworks, fc3_bwDworksSize, &beta, fc2_Tensor, dev_bwd_fc3_Output);

			//fc2 back
			cudnnActivationBackward(cudnn, fc2_Act_Desc, &alpha, fc2_Tensor, dev_fc2_Actout, fc2_Tensor,
				dev_bwd_fc3_Output, fc2_Tensor, dev_fc2_Output, &beta, fc2_Tensor, dev_bw_fc2_ActDelta);
			cudnnConvolutionBackwardBias(cudnn, &alpha, fc2_Tensor, dev_bw_fc2_ActDelta, &beta, fc2_biasTensor, dev_bw_fc2Bias);
			cudnnConvolutionBackwardFilter(cudnn, &alpha, fc1_Tensor, dev_fc1_Actout, fc2_Tensor, dev_bw_fc2_ActDelta, fc2_Desc,
				fc2_bwFAlgo, dev_fc2_bwFworks, fc2_bwFworksSize, &beta, fc2_Filter_Desc, dev_bwf_fc2_Output);
			cudnnConvolutionBackwardData(cudnn, &alpha, fc2_Filter_Desc, dev_fc2_Filter, fc2_Tensor, dev_bw_fc2_ActDelta,
				fc2_Desc, fc2_bwDAlgo, dev_fc2_bwDworks, fc2_bwDworksSize, &beta, fc1_Tensor, dev_bwd_fc2_Output);


			//fc1 back
			cudnnActivationBackward(cudnn, fc1_Act_Desc, &alpha, fc1_Tensor, dev_fc1_Actout, fc1_Tensor, dev_bwd_fc2_Output, fc1_Tensor, dev_fc1_Output, &beta, fc1_Tensor, dev_bw_fc1_ActDelta);
			cudnnConvolutionBackwardBias(cudnn, &alpha, fc1_Tensor, dev_bw_fc1_ActDelta, &beta, fc1_biasTensor, dev_bw_fc1Bias);
			cudnnConvolutionBackwardFilter(cudnn, &alpha, pool5_Tensor, dev_conv5pool_Act_output, fc1_Tensor, dev_bw_fc1_ActDelta, fc1_Desc, fc1_bwFAlgo, dev_fc1_bwFworks, fc1_bwFworksSize, &beta, fc1_Filter_Desc, dev_bwf_fc1_Output);
			cudnnConvolutionBackwardData(cudnn, &alpha, fc1_Filter_Desc, dev_fc1_Filter, fc1_Tensor, dev_bw_fc1_ActDelta, fc1_Desc, fc1_bwDAlgo, dev_fc1_bwDworks, fc1_bwDworksSize, &beta, pool5_Tensor, dev_bwd_fc1_Output);


			//conv5 back
			cudnnActivationBackward(cudnn, conv5pool_Act_Desc, &alpha, pool5_Tensor, dev_conv5pool_Act_output, pool5_Tensor, dev_bwd_fc1_Output, pool5_Tensor, dev_pool5_Output, &beta, pool5_Tensor, dev_bw_conv5pool_ActDelta);
			cudnnPoolingBackward(cudnn, pool5_Desc, &alpha, pool5_Tensor, dev_pool5_Output, pool5_Tensor, dev_bw_conv5pool_ActDelta, conv5_Tensor, dev_conv5_Output, &beta, conv5_Tensor, dev_pool5_Delta);
			cudnnActivationBackward(cudnn, conv5_Act_Desc, &alpha, conv5_Tensor, dev_conv5Act_output, conv5_Tensor, dev_pool5_Delta, conv5_Tensor, dev_conv5_Output, &beta, conv5_Tensor, dev_bw_conv5_ActDelta);
			cudnnConvolutionBackwardBias(cudnn, &alpha, conv5_Tensor, dev_bw_conv5_ActDelta, &beta, conv5_biasTensor, dev_bw_conv5Bias);
			cudnnConvolutionBackwardFilter(cudnn, &alpha, conv4_Tensor, dev_conv4Act_output, conv5_Tensor, dev_bw_conv5_ActDelta, conv5_Desc, conv5_bwFAlgo, dev_conv5_bwFworks, conv5_bwFworksSize, &beta, conv5_Filter_Desc, dev_bwf_conv5_Output);
			cudnnConvolutionBackwardData(cudnn, &alpha, conv5_Filter_Desc, dev_conv5_Filter, conv5_Tensor, dev_bw_conv5_ActDelta, conv5_Desc, conv5_bwDAlgo, dev_conv5_bwDworks, conv5_bwDworksSize, &beta, conv4_Tensor, dev_bwd_conv5_Output);

			//conv4 back
			
			cudnnActivationBackward(cudnn, conv4_Act_Desc, &alpha, conv4_Tensor, dev_conv4Act_output, conv4_Tensor, dev_bwd_conv5_Output, conv4_Tensor, dev_conv4_Output, &beta, conv4_Tensor, dev_bw_conv4_ActDelta);
			cudnnConvolutionBackwardBias(cudnn, &alpha, conv4_Tensor, dev_bw_conv4_ActDelta, &beta, conv4_biasTensor, dev_bw_conv4Bias);
			cudnnConvolutionBackwardFilter(cudnn, &alpha, conv3_Tensor, dev_conv3Act_output, conv4_Tensor, dev_bw_conv4_ActDelta, conv4_Desc, conv4_bwFAlgo, dev_conv4_bwFworks, conv4_bwFworksSize, &beta, conv4_Filter_Desc, dev_bwf_conv4_Output);
			cudnnConvolutionBackwardData(cudnn, &alpha, conv4_Filter_Desc, dev_conv4_Filter, conv4_Tensor, dev_bw_conv4_ActDelta, conv4_Desc, conv4_bwDAlgo, dev_conv4_bwDworks, conv4_bwDworksSize, &beta, conv3_Tensor, dev_bwd_conv4_Output);

			//conv3 back
			cudnnActivationBackward(cudnn, conv3_Act_Desc, &alpha, conv3_Tensor, dev_conv3Act_output, conv3_Tensor, dev_bwd_conv4_Output, conv3_Tensor, dev_conv3_Output, &beta, conv3_Tensor, dev_bw_conv3_ActDelta);
			cudnnConvolutionBackwardBias(cudnn, &alpha, conv3_Tensor, dev_bw_conv3_ActDelta, &beta, conv3_biasTensor, dev_bw_conv3Bias);
			cudnnConvolutionBackwardFilter(cudnn, &alpha, pool2_Tensor, dev_conv2pool_Act_output, conv3_Tensor, dev_bw_conv3_ActDelta, conv3_Desc, conv3_bwFAlgo, dev_conv3_bwFworks, conv3_bwFworksSize, &beta, conv3_Filter_Desc, dev_bwf_conv3_Output);
			cudnnConvolutionBackwardData(cudnn, &alpha, conv3_Filter_Desc, dev_conv3_Filter, conv3_Tensor, dev_bw_conv3_ActDelta, conv3_Desc, conv3_bwDAlgo, dev_conv3_bwDworks, conv3_bwDworksSize, &beta, pool2_Tensor, dev_bwd_conv3_Output);

			//conv2 back
			cudnnActivationBackward(cudnn, conv2pool_Act_Desc, &alpha, pool2_Tensor, dev_conv2pool_Act_output, pool2_Tensor, dev_bwd_conv3_Output, pool2_Tensor, dev_pool2_Output, &beta, pool2_Tensor, dev_bw_conv2pool_ActDelta);
			cudnnPoolingBackward(cudnn, pool2_Desc, &alpha, pool2_Tensor, dev_pool2_Output, pool2_Tensor, dev_bw_conv2pool_ActDelta, conv2_Tensor, dev_conv2_Output, &beta, conv2_Tensor, dev_pool2_Delta);
			cudnnActivationBackward(cudnn, conv2_Act_Desc, &alpha, conv2_Tensor, dev_conv2Act_output, conv2_Tensor, dev_pool2_Delta, conv2_Tensor, dev_conv2_Output, &beta, conv2_Tensor, dev_bw_conv2_ActDelta);
			cudnnConvolutionBackwardBias(cudnn, &alpha, conv2_Tensor, dev_bw_conv2_ActDelta, &beta, conv2_biasTensor, dev_bw_conv2Bias);
			cudnnConvolutionBackwardFilter(cudnn, &alpha, pool1_Tensor, dev_conv1pool_Act_output, conv2_Tensor, dev_bw_conv2_ActDelta, conv2_Desc, conv2_bwFAlgo, dev_conv2_bwFworks, conv2_bwFworksSize, &beta, conv2_Filter_Desc, dev_bwf_conv2_Output);
			cudnnConvolutionBackwardData(cudnn, &alpha, conv2_Filter_Desc, dev_conv2_Filter, conv2_Tensor, dev_bw_conv2_ActDelta, conv2_Desc, conv2_bwDAlgo, dev_conv2_bwDworks, conv2_bwDworksSize, &beta, pool1_Tensor, dev_bwd_conv2_Output);

			//conv1 back
			cudnnActivationBackward(cudnn, conv1pool_Act_Desc, &alpha, pool1_Tensor, dev_conv1pool_Act_output, pool1_Tensor, dev_bwd_conv2_Output, pool1_Tensor, dev_pool1_Output, &beta, pool1_Tensor, dev_bw_conv1pool_ActDelta); 
			cudnnPoolingBackward(cudnn, pool1_Desc, &alpha, pool1_Tensor, dev_pool1_Output, pool1_Tensor, dev_bw_conv1pool_ActDelta, conv1_Tensor, dev_conv1_Output, &beta, conv1_Tensor, dev_pool1_Delta);
			cudnnActivationBackward(cudnn, conv1_Act_Desc, &alpha, conv1_Tensor, dev_conv1Act_output, conv1_Tensor, dev_pool1_Delta, conv1_Tensor, dev_conv1_Output, &beta, conv1_Tensor, dev_bw_conv1_ActDelta);
			cudnnConvolutionBackwardBias(cudnn, &alpha, conv1_Tensor, dev_bw_conv1_ActDelta, &beta, conv1_biasTensor, dev_bw_conv1Bias);
			cudnnConvolutionBackwardFilter(cudnn, &alpha, input_Tensor, dev_input, conv1_Tensor, dev_bw_conv1_ActDelta, conv1_Desc, conv1_bwFAlgo, dev_conv1_bwFworks, conv1_bwFworksSize, &beta, conv1_Filter_Desc, dev_bwf_conv1_Output);



			//learning_rate = static_cast<float>(learning_rate * pow((1.0 + 0.0001*iter), 0.75));


			// Fully connected 3
			cublasSaxpy(cublasHandle, static_cast<int>(lastLayer_numOut * fc2FilterCnt * 1 * 1), &learning_rate, dev_bwf_fc3_Output, 1, dev_fc3_Filter, 1);
			cublasSaxpy(cublasHandle, static_cast<int>(1 * lastLayer_numOut * 1 * 1), &learning_rate, dev_bw_fc3Bias, 1, dev_fc3_Bias, 1);

			// Fully connected 2
			cublasSaxpy(cublasHandle, static_cast<int>(fc2FilterCnt * fc1FilterCnt * 1 * 1), &learning_rate, dev_bwf_fc2_Output, 1, dev_fc2_Filter, 1);
			cublasSaxpy(cublasHandle, static_cast<int>(1 * fc2FilterCnt * 1 * 1), &learning_rate, dev_bw_fc2Bias, 1, dev_fc2_Bias, 1);

			// Fully connected 1
			cublasSaxpy(cublasHandle, static_cast<int>(fc1FilterCnt * conv5FilterCnt * pool2OutHeight * pool2OutWidth), &learning_rate, dev_bwf_fc1_Output, 1, dev_fc1_Filter, 1);
			cublasSaxpy(cublasHandle, static_cast<int>(1 * fc1FilterCnt * 1 * 1), &learning_rate, dev_bw_fc1Bias, 1, dev_fc1_Bias, 1);



			// Conv5
			cublasSaxpy(cublasHandle, static_cast<int>(conv5FilterCnt * conv4FilterCnt * conv5FilterHeight * conv5FilterWidth), &learning_rate, dev_bwf_conv5_Output, 1, dev_conv5_Filter, 1);
			cublasSaxpy(cublasHandle, static_cast<int>(1 * conv5FilterCnt * 1 * 1), &learning_rate, dev_bw_conv5Bias, 1, dev_conv5_Bias, 1);

			// Conv4
			cublasSaxpy(cublasHandle, static_cast<int>(conv4FilterCnt * conv3FilterCnt * conv4FilterHeight * conv4FilterWidth), &learning_rate, dev_bwf_conv4_Output, 1, dev_conv4_Filter, 1);
			cublasSaxpy(cublasHandle, static_cast<int>(1 * conv4FilterCnt * 1 * 1), &learning_rate, dev_bw_conv4Bias, 1, dev_conv4_Bias, 1);

			// Conv3
			cublasSaxpy(cublasHandle, static_cast<int>(conv3FilterCnt * conv2FilterCnt * conv3FilterHeight * conv3FilterWidth), &learning_rate, dev_bwf_conv3_Output, 1, dev_conv3_Filter, 1);
			cublasSaxpy(cublasHandle, static_cast<int>(1 * conv3FilterCnt * 1 * 1), &learning_rate, dev_bw_conv3Bias, 1, dev_conv3_Bias, 1);


			// Conv2
			cublasSaxpy(cublasHandle, static_cast<int>(conv2FilterCnt * conv1FilterCnt * conv2FilterHeight * conv2FilterWidth), &learning_rate, dev_bwf_conv2_Output, 1, dev_conv2_Filter, 1);
			cublasSaxpy(cublasHandle, static_cast<int>(1 * conv2FilterCnt * 1 * 1), &learning_rate, dev_bw_conv2Bias, 1, dev_conv2_Bias, 1);

			// Conv1
			cublasSaxpy(cublasHandle, static_cast<int>(conv1FilterCnt * input_channelCnt * conv1FilterHeight * conv1FilterWidth), &learning_rate, dev_bwf_conv1_Output, 1, dev_conv1_Filter, 1);
			cublasSaxpy(cublasHandle, static_cast<int>(1 * conv1FilterCnt * 1 * 1), &learning_rate, dev_bw_conv1Bias, 1, dev_conv1_Bias, 1);



			//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

			//accuracy 계산

			for (size_t i = 0; i < batchSize; i++) {
				if (predicted[i] == target_train_batch[i])
					count++;
			}


		}



		cout << "# EPOCH (" << setw(3) << iter + 1 << " / " << epoch << " ) , " << "Train Set Accuracy ( " << setw(5) << (count*100.0 / ImageNum) << " %)" << " , 맞은 갯수 :: " << setw(5) << count << endl;

	}


	endTime_train = clock();


	///////////////////////////////////////////
	//test 계산용 데이터 셋 준비 


	vector<string> LabelBox_test; // 라벨 정리를 위해 생성
	vector<pair<int, string>> LabelTable_test; // 라벨링 마다 넘버 부여
	float* target_test = new float[ImageNum_test]; // target 값 , 라벨에 따른 지정된 넘버 값이 담긴 배열


												   // 라벨에 번호 부여를 위해 LabelBox 벡터에 값 복사 하고 정렬 및 중복 삭제
	for (int i = 0; i < ImageNum_test; i++) {
		LabelBox_test.push_back(ImgBox_test[i].second);
		//std::cout<< "라벨 출력 :: " << ImgBox[i].second << std::endl; // 입력받은순서대로 라벨 출력 -> 예시 "라벨 출력 :: automobile"
	}

	sort(LabelBox_test.begin(), LabelBox_test.end());
	LabelBox_test.erase(unique(LabelBox_test.begin(), LabelBox_test.end()), LabelBox_test.end());
	int nLabelBoxSize_test = LabelBox_test.size();

	// 라벨 번호 부여
	for (int i = 0; i < nLabelBoxSize_test; i++) {
		LabelTable_test.push_back({ { i },{ LabelBox_test[i] } });
		//std::cout << "LabelBox :: " << LabelBox[i] << std::endl;// -> 예시 "LabelBox :: truck"
	}

	//target 셋팅
	for (int i = 0; i < ImageNum_test; i++) {
		for (int j = 0; j < LabelTable_test.size(); j++) {
			if (ImgBox_test[i].second == LabelTable_test[j].second) {
				target_test[i] = LabelTable_test[j].first;
			}
		}
	}


	//입력변수
	float* target_test_batch = new float[batch_size_test];

	float* Input_test = new float[batch_size_test * input_channelCnt * imageHeight * imageWidth];

	float* yhat = new float[batch_size_test* num_labels];// soft max 결과 값 

	int* predicted = new int[ImageNum_test]; // 소프트 맥스 결과 값에서 가장 큰 값이 들어 있는 위치(인덱스)를 저장하는 배열
	int count = 0;

	float* dev_input_test;
	checkCudaErrors(cudaMalloc((void**)&dev_input_test, sizeof(float) * batch_size_test * input_channelCnt * imageHeight * imageWidth));

	for (int a = 0; a < ImageNum_test / batch_size_test; a++) { // 배치 계산 루프


		for (int i = 0; i < batch_size_test; i++) {
			unsigned char* temp_test = ImgBox_test[i + (batch_size_test * a)].first.data;

			for (int c = 0; c < input_channelCnt; c++) {
				for (int y = 0; y < imageHeight; y++) {
					for (int x = 0; x < imageWidth; x++) {
						Input_test[i * input_channelCnt * imageHeight * imageWidth + c * imageHeight * imageWidth + y * imageWidth + x] = temp_test[input_channelCnt * imageHeight * x + input_channelCnt * y + c] / 255.0;
					}
				}
			}
		}
		for (int i = 0; i < batch_size_test; i++) {
			target_test_batch[i] = target_test[i + (batch_size_test * a)];
		}


		checkCudaErrors(cudaMemcpy(dev_input_test, Input_test, sizeof(float) * batch_size_test * input_channelCnt * imageHeight * imageWidth, cudaMemcpyHostToDevice));
		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		//GPU test 계산 들어갈 자리
		//conv1 forward
		cudnnConvolutionForward(cudnn, &alpha, input_Tensor, dev_input, conv1_Filter_Desc, dev_conv1_Filter, conv1_Desc, conv1_fwAlgo, dev_conv1_works, conv1_worksSize, &beta, conv1_Tensor, dev_conv1_Output);
		cudnnAddTensor(cudnn, &alpha, conv1_biasTensor, dev_conv1_Bias, &alpha, conv1_Tensor, dev_conv1_Output);
		cudnnActivationForward(cudnn, conv1_Act_Desc, &alpha, conv1_Tensor, dev_conv1_Output, &beta, conv1_Tensor, dev_conv1Act_output);
		cudnnPoolingForward(cudnn, pool1_Desc, &alpha, conv1_Tensor, dev_conv1Act_output, &beta, pool1_Tensor, dev_pool1_Output);
		cudnnActivationForward(cudnn, conv1pool_Act_Desc, &alpha, pool1_Tensor, dev_pool1_Output, &beta, pool1_Tensor, dev_conv1pool_Act_output);
		//conv2 forward
		cudnnConvolutionForward(cudnn, &alpha, pool1_Tensor, dev_conv1pool_Act_output, conv2_Filter_Desc, dev_conv2_Filter, conv2_Desc, conv2_fwAlgo, dev_conv2_works, conv2_worksSize, &beta, conv2_Tensor, dev_conv2_Output);
		cudnnAddTensor(cudnn, &alpha, conv2_biasTensor, dev_conv2_Bias, &alpha, conv2_Tensor, dev_conv2_Output);
		cudnnActivationForward(cudnn, conv2_Act_Desc, &alpha, conv2_Tensor, dev_conv2_Output, &beta, conv2_Tensor, dev_conv2Act_output);
		cudnnPoolingForward(cudnn, pool2_Desc, &alpha, conv2_Tensor, dev_conv2Act_output, &beta, pool2_Tensor, dev_pool2_Output);
		cudnnActivationForward(cudnn, conv2pool_Act_Desc, &alpha, pool2_Tensor, dev_pool2_Output, &beta, pool2_Tensor, dev_conv2pool_Act_output);
		//conv3 forward
		cudnnConvolutionForward(cudnn, &alpha, pool2_Tensor, dev_conv2pool_Act_output, conv3_Filter_Desc, dev_conv3_Filter, conv3_Desc, conv3_fwAlgo, dev_conv3_works, conv3_worksSize, &beta, conv3_Tensor, dev_conv3_Output);
		cudnnAddTensor(cudnn, &alpha, conv3_biasTensor, dev_conv3_Bias, &alpha, conv3_Tensor, dev_conv3_Output);
		cudnnActivationForward(cudnn, conv3_Act_Desc, &alpha, conv3_Tensor, dev_conv3_Output, &beta, conv3_Tensor, dev_conv3Act_output);
		//conv4 forward
		cudnnConvolutionForward(cudnn, &alpha, conv3_Tensor, dev_conv3Act_output, conv4_Filter_Desc, dev_conv4_Filter, conv4_Desc, conv4_fwAlgo, dev_conv4_works, conv4_worksSize, &beta, conv4_Tensor, dev_conv4_Output);
		cudnnAddTensor(cudnn, &alpha, conv4_biasTensor, dev_conv4_Bias, &alpha, conv4_Tensor, dev_conv4_Output);
		cudnnActivationForward(cudnn, conv4_Act_Desc, &alpha, conv4_Tensor, dev_conv4_Output, &beta, conv4_Tensor, dev_conv4Act_output);
		//conv5 forward
		cudnnConvolutionForward(cudnn, &alpha, conv4_Tensor, dev_conv4Act_output, conv5_Filter_Desc, dev_conv5_Filter, conv5_Desc, conv5_fwAlgo, dev_conv5_works, conv5_worksSize, &beta, conv5_Tensor, dev_conv5_Output);
		cudnnAddTensor(cudnn, &alpha, conv5_biasTensor, dev_conv5_Bias, &alpha, conv5_Tensor, dev_conv5_Output);
		cudnnActivationForward(cudnn, conv5_Act_Desc, &alpha, conv5_Tensor, dev_conv5_Output, &beta, conv5_Tensor, dev_conv5Act_output);
		cudnnPoolingForward(cudnn, pool5_Desc, &alpha, conv5_Tensor, dev_conv5Act_output, &beta, pool5_Tensor, dev_pool5_Output);
		cudnnActivationForward(cudnn, conv5pool_Act_Desc, &alpha, pool5_Tensor, dev_pool5_Output, &beta, pool5_Tensor, dev_conv5pool_Act_output);
		//fc1 forward
		cudnnConvolutionForward(cudnn, &alpha, pool5_Tensor, dev_conv5pool_Act_output, fc1_Filter_Desc, dev_fc1_Filter, fc1_Desc, fc1_fwAlgo, dev_fc1_works, fc1_worksSize, &beta, fc1_Tensor, dev_fc1_Output);
		cudnnAddTensor(cudnn, &alpha, fc1_biasTensor, dev_fc1_Bias, &alpha, fc1_Tensor, dev_fc1_Output);
		cudnnActivationForward(cudnn, fc1_Act_Desc, &alpha, fc1_Tensor, dev_fc1_Output, &beta, fc1_Tensor, dev_fc1_Actout);
		//fc2 forward
		cudnnConvolutionForward(cudnn, &alpha, fc1_Tensor, dev_fc1_Actout, fc2_Filter_Desc, dev_fc2_Filter, fc2_Desc, fc2_fwAlgo, dev_fc2_works, fc2_worksSize, &beta, fc2_Tensor, dev_fc2_Output);
		cudnnAddTensor(cudnn, &alpha, fc2_biasTensor, dev_fc2_Bias, &alpha, fc2_Tensor, dev_fc2_Output);
		cudnnActivationForward(cudnn, fc2_Act_Desc, &alpha, fc2_Tensor, dev_fc2_Output, &beta, fc2_Tensor, dev_fc2_Actout);
		//fc3 forward
		cudnnConvolutionForward(cudnn, &alpha, fc2_Tensor, dev_fc2_Actout, fc3_Filter_Desc, dev_fc3_Filter, fc3_Desc, fc3_fwAlgo, dev_fc3_works, fc3_worksSize, &beta, fc3_Tensor, dev_fc3_Output);
		cudnnAddTensor(cudnn, &alpha, fc3_biasTensor, dev_fc3_Bias, &alpha, fc3_Tensor, dev_fc3_Output);
		cudnnSoftmaxForward(cudnn, sftAlgo, sftMode, &alpha, fc3_Tensor, dev_fc3_Output, &beta, fc3_Tensor, dev_smaxOutput);
		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		cudaMemcpy(yhat, dev_smaxOutput, sizeof(float) * batch_size_test * num_labels, cudaMemcpyDeviceToHost);

		//one hot 위치 찾기


		for (size_t i = 0; i < batch_size_test; i++) {
			float temp = yhat[i * 10];// 예측된 첫번째 값을 임시 변수에 저장
			int indexJ = 0; // 가장 큰값을 찾기 위해 사용 되는 위치 인덱스

			for (size_t j = 0; j < num_labels - 1; j++) {
				if (temp > yhat[i * 10 + j + 1]) // 임시 변수에 넣어준 값과 비교
				{
					yhat[i * 10 + j + 1] = 0; // 임시 변수에 들어 있는 값보다 작다면 0 입력
				}
				else                      // 임시 변수에 들어 있는 값보다 크다면
				{
					temp = yhat[i * 10 + j + 1]; // 임시 변수에 해당 값을 저장
					yhat[i * 10 + indexJ] = 0; // 임시 변수에 이전에 들어 있던 값의 인덱스를 이용하여 이전 값 위치에 0 입력
					indexJ = j + 1; // 제일 큰 값(정답 예측 값) 위치를 인덱스 변수에 저장
				}
			}

			predicted[i] = indexJ; // 해당 이미지의 소프트 맥스 값중 가장 큰 값을 갖는 위치 인덱스를 배열에 저장
		}

		//accuracy 계산

		for (size_t i = 0; i < batch_size_test; i++) {
			if (predicted[i] == target_test_batch[i])
				count++;
		}


	}


	std::cout << "================== 학습 정보=================" << endl;

	cout << " 학습 계산 시간             :: " << setw(10) << ((endTime_train - startTime_train) / ((CLOCKS_PER_SEC) * 60)) << " 분" << endl;
	cout << " 학습에 사용된 이미지 수    :: " << setw(10) << ImageNum << " 개" << endl;
	cout << " 배치 크기(batch size)      :: " << setw(10) << batchSize << " 개" << endl;
	cout << " 학습 회수(epoch)           :: " << setw(10) << epoch << " 회" << endl;
	cout << " 학습률 (learning_rate)     :: " << setw(10) << learning_rate << endl;


	cout << endl; cout << endl;

	std::cout << "=============== 테스트 정보 ===============" << endl;
	cout << " 테스트에 사용된 이미지 수   :: " << setw(10) << ImageNum_test << " 개" << endl;
	cout << " 배치 크기(batch size)       :: " << setw(10) << batch_size_test << " 회" << endl;
	cout << endl; cout << endl;

	std::cout << "=============== 테스트 결과 ===============" << endl;
	cout << setw(2) << ImageNum_test << " 개의 이미지들 중 " << count << " 개 맞음" << endl;
	cout << "정확도(Accuracy)     :: " << (count*1.0 / ImageNum_test) * 100 << " %" << endl;
	cout << "에러(Error)          :: " << (1 - (count*1.0 / ImageNum_test)) * 100 << " %" << endl;
	cout << endl; cout << endl;

	std::cout << "끝==========================================" << endl;

	cudaFree(dev_bwf_conv1_Output);
	cudaFree(dev_bw_conv1Bias);
	cudaFree(dev_bw_conv1_ActDelta);
	cudaFree(dev_pool1_Delta);
	cudaFree(dev_bwd_fc3_Output);

	cudaFree(dev_bwf_fc3_Output);
	cudaFree(dev_bw_fc3Bias);


	cudaFree(dev_smaxOutput);
	cudaFree(dev_fc3_Bias);

	cudaFree(dev_fc3_Output);
	cudaFree(dev_fc3_Filter);
	cudaFree(dev_pool1_Output);
	cudaFree(dev_conv1Act_output);
	cudaFree(dev_conv1_Bias);

	cudaFree(dev_conv1_Output);
	cudaFree(dev_conv1_Filter);
	cudaFree(dev_input);



}