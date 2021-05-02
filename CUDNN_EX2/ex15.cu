

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
//**���Ŀ� �� �������� GPU �޸� SizeȮ�� �ʿ�**
//***********************************************

//***********************
//**�ɼ� ���� ��� �߰�**
//***********************

//********************
//**alpha, beta ����**
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
	const int numImgs = 100; // �̹��� �� ����
	string folder_path = "D:\\DataSet\\cifar\\test"; // �̹����� ����Ǿ� �ִ� ���� ���
	vector<pair<Mat, string>> ImgBox; // �̹��� ������, �̹��� �̸�
	ImgBox = TraverseFilesUsingDFS(folder_path);
	vector<string> LabelBox; // �� ������ ���� ����
	vector<pair<int, string>> LabelTable; // �󺧸� ���� �ѹ� �ο�
	vector<pair<Mat, int>> ImgBox2; // �̹��� ������, �� �ѹ�
	vector<float> labels(numImgs);
	vector<vector<int>> TargetY; // �� �ѹ� -> ������ ���� �����ͷ� ����



								 //���� ����
	float Filter[3][3][8][8];

	float sigma = sqrt(2 / (float)(3 + 3));

	random_device rd;
	mt19937 gen(rd());
	normal_distribution<float> d(0, sigma);


	//Weights ����
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


	//Weights ����
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


	// �󺧿� ��ȣ �ο��� ���� LabelBox ���Ϳ� �� ���� �ϰ� ���� �� �ߺ� ����
	for (int i = 0; i < numImgs; i++)
	{
		LabelBox.push_back(ImgBox[i].second);
	}

	sort(LabelBox.begin(), LabelBox.end());
	LabelBox.erase(unique(LabelBox.begin(), LabelBox.end()), LabelBox.end());
	size_t nLabelBoxSize = LabelBox.size();

	// �� ��ȣ �ο�
	for (int i = 0; i < nLabelBoxSize; i++)
	{
		LabelTable.push_back({ { i },{ LabelBox[i] } });
	}

	//ImgBox2 ����
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


	// TargetY ����, ���� ������ ���·� ǥ��
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




	// 4�� ��� ���� �Ҵ� ����.
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

	// mat ���� - > 4�� ���
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






	// �̹��� ������ �غ� ��

	//**********
	//**Handle**
	//**********
	cudnnHandle_t cudnn;
	cudnnCreate(&cudnn);
	cublasHandle_t cublasHandle;
	cublasCreate(&cublasHandle);

	//********
	//**����**
	//********

	//�Էº���
	const int ImageNum = 1;
	const int FeatureNum = 3;
	const int FeatureHeight = 32;
	const int FeatureWidth = 32;

	//********
	//**�Է�**
	//********
	//GPU�� �Է���� �޸� �Ҵ� �� �� ����

	float* dev_Input;
	cudaMalloc((void**)&dev_Input, sizeof(Input));
	cudaMemcpy(dev_Input, Input, sizeof(Input), cudaMemcpyHostToDevice);//�Է���� ����ü ����, �Ҵ�, �ʱ�ȭ
	cudnnTensorDescriptor_t in_desc; //�Է� ������ �� ������ ���� �ִ� ����ü�� ����Ű�� ���� ������
	cudnnCreateTensorDescriptor(&in_desc); // 4D tensor ����ü ��ü ����

										   // 4D tensor ����ü �ʱ�ȭ �Լ�
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
	//**Feedforward �������**
	//************************
	//************************

	//*******************
	//**Convolution����**
	//*******************

	//���� ������ ���� - ���� ���� �����ϵ���
	const int filt_n = 3;
	const int filt_c = 3;
	const int filt_h = 8;
	const int filt_w = 8;
	/*
	//���� ����
	float Filter[filt_n][filt_c][filt_h][filt_w];

	//���� ����
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

	//GPU�� ������� ����
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


	//���ͱ���ü ����, ����, �ʱ�ȭ
	cudnnFilterDescriptor_t filt_desc; // ���� ������ ���� ����ü�� ����Ű�� ���� ������
	cudnnCreateFilterDescriptor(&filt_desc); // ���� ����ü ����
	cudnnSetFilter4dDescriptor( // 4d filter ����ü ��ü �ʱ�ȭ
		/*filterDesc,*/filt_desc,
		/*dataType,*/CUDNN_DATA_FLOAT,
		/*format,*/CUDNN_TENSOR_NCHW,
		/*Number of output feature maps*/filt_n,
		/*Number of input feature maps.*/filt_c,
		/*Height of each filter.*/filt_h,
		/*Width of each filter.*/filt_w);

	//Convolution ���꿡���� ���� ���� - ���� ���� �����ϵ���
	const int pad_h = 2; //padding ����
	const int pad_w = 2; //padding ����
	const int str_h = 4; //stride ����
	const int str_w = 4; //stride ����
	const int dil_h = 1; //dilated ����
	const int dil_w = 1; //dilated ����

						 //Convolution ����ü ���� �� �Ҵ�
	cudnnConvolutionDescriptor_t conv_desc; // Convolution ������ ���� ������ ���� ����ü ������
	cudnnCreateConvolutionDescriptor(&conv_desc); // Convolution ����ü ��ü ����
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

	//Convolution ��� ������� ���� �� �Ҵ�
	cudnnTensorDescriptor_t out_conv_desc;
	cudnnCreateTensorDescriptor(&out_conv_desc);
	//Convolution ������ ������ ����
	int out_conv_n;
	int out_conv_c;
	int out_conv_h;
	int out_conv_w;
	cudnnGetConvolution2dForwardOutputDim( // �־��� ����, tensor, convolution ����ü ������ ��������, 2D convolution ��꿡 ���� 4d tensor�� ��� ���� ������ ��ȯ, �� ��� output�� ���� �� �ε�...
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

	//Convolution��� ����
	//float Output_Conv[ImageNum][FeatureNum][outputDimHW][outputDimHW];


	//GPU�� Convolution ��� ��� �Ҵ�
	float* dev_Output_Conv;
	cudaMalloc((void**)&dev_Output_Conv, sizeof(float) * out_conv_c * out_conv_h * out_conv_n * out_conv_w);
	//Convolution ����ü �ʱ�ȭ
	cudnnSetTensor4dDescriptor(out_conv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, ImageNum, FeatureNum, outputDim, outputDim);
	//�Է°� ����, ������� �е�, ��Ʈ���̵尡 ���� ���� �־������� ���� ���� �˰����� ���������� �˾Ƴ���
	cudnnConvolutionFwdAlgo_t alg;
	alg = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
	//Conv ���� ������ũ�� �˾Ƴ��� �� �� ���� �޸� �Ҵ� �߰�
	size_t WS_size = 0;
	cudnnGetConvolutionForwardWorkspaceSize(// This function returns the amount of GPU memory workspace
		cudnn, in_desc, filt_desc, conv_desc, out_conv_desc, alg, &WS_size);
	size_t* dev_WS;
	cudaMalloc((void**)&dev_WS, WS_size);
	//����
	float alpha = 1.0;
	float beta = 0.0;

	float Output_Conv[ImageNum][FeatureNum][outputDimHW][outputDimHW];

	//Convolution��� GPU�� ����


	//********
	//**Bias**
	//********
	beta = 1.0f;
	//Bias ��� ������� ����
	//float Output_Bias[ImageNum][FeatureNum][outputDimHW][outputDimHW];
	//bias �� ����
	//float biasValue[filt_n] = { -10.0f };


	std::vector<float> biasValue;
	biasValue.resize(filt_n);
	InitWeightsbias(filt_n, biasValue);

	//GPU�� bias�� ����
	float* dev_Bias;
	cudaMalloc((void**)&dev_Bias, sizeof(float));
	cudaMemcpy(dev_Bias, &biasValue[0], sizeof(float), cudaMemcpyHostToDevice);
	//bias��� ������� ����, �Ҵ�
	cudnnTensorDescriptor_t bias_desc;
	cudnnCreateTensorDescriptor(&bias_desc);
	cudnnSetTensor4dDescriptor(bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, filt_n, 1, 1);
	//bias ���� ����

	//Bias�� ���
	//cudaMemcpy(Output_Bias, dev_Output_Conv, sizeof(float) * ImageNum * FeatureNum * outputDimHW * outputDimHW, cudaMemcpyDeviceToHost);

	//***********************
	//**Actiovation Funtion**
	//***********************
	beta = 0.0;
	//Activation Function ����ü ���� �� �Ҵ�
	cudnnActivationDescriptor_t act_desc;
	cudnnCreateActivationDescriptor(&act_desc);
	//Activation Function ���� ���� - ���� ���������ϵ���
	cudnnActivationMode_t Activation_Function;
	//Activation_Function = CUDNN_ACTIVATION_RELU;
	//Activation_Function = CUDNN_ACTIVATION_TANH;
	Activation_Function = CUDNN_ACTIVATION_SIGMOID;
	cudnnSetActivationDescriptor(act_desc, Activation_Function, CUDNN_PROPAGATE_NAN, 0);
	//Activation Function �޸� GPU�� ����
	float* dev_Output_Act;
	cudaMalloc((void**)&dev_Output_Act, sizeof(float) * outputDimHW * outputDimHW * 3 * 3);
	//Activatin Function �������


	//Activation Function ����� ���� ���
	//float Output_Activation[ImageNum][FeatureNum][outputDimHW][outputDimHW];
	//cudaMemcpy(Output_Activation, dev_Output_Act, sizeof(float) * ImageNum * FeatureNum * outputDimHW * outputDimHW, cudaMemcpyDeviceToHost);
	//Actavation Function ���

	//***************
	//**Pooling����**
	//***************
	//Pooling ���꿡�� ���� ���� - ���� ���� �����ϵ���
	beta = 0.0;
	const int pool_wind_h = 2;
	const int pool_wind_w = 2;
	const int pool_pad_h = 0;
	const int pool_pad_w = 0;
	const int pool_strd_w = 2;
	const int pool_strd_h = 2;
	//Pooling ����ü ���� �� �Ҵ� - ���� Pooling ��� ���� �����ϵ���
	cudnnPoolingDescriptor_t pool_desc;
	cudnnCreatePoolingDescriptor(&pool_desc);
	cudnnSetPooling2dDescriptor(pool_desc, CUDNN_POOLING_MAX, CUDNN_PROPAGATE_NAN,
		pool_wind_h, pool_wind_w, pool_pad_h, pool_pad_w, pool_strd_h, pool_strd_w);
	//Pooling ������� ��� ���� �� �Ҵ�
	cudnnTensorDescriptor_t out_pool_desc;
	cudnnCreateTensorDescriptor(&out_pool_desc);
	//Pooling ������ ������
	int out_pool_n;
	int out_pool_c;
	int out_pool_h;
	int out_pool_w;
	//Pooling ������ ������ ����
	cudnnGetPooling2dForwardOutputDim(pool_desc, out_conv_desc, &out_pool_n, &out_pool_c, &out_pool_h, &out_pool_w);
	//GPU�� Pooling ������ �޸��Ҵ�
	float* dev_Output_Pool;
	cudaMalloc((void**)&dev_Output_Pool, sizeof(float) * out_pool_n * out_pool_c * out_pool_h * out_pool_w);
	//Pooling ������� ����ü �ʱ�ȭ
	cudnnSetTensor4dDescriptor(out_pool_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, out_pool_n, out_pool_c, out_pool_h, out_pool_w);
	//Pooling���� ����

	//Pooling���
	//float Output_Pool[ImageNum][FeatureNum][(outputDimHW + 2 * pool_pad_h) / pool_strd_h][(outputDimHW + 2 * pool_pad_w) / pool_strd_w];
	//cudaMemcpy(Output_Pool, dev_Output_Pool, sizeof(float) * out_pool_n * out_pool_c * out_pool_h * out_pool_w, cudaMemcpyDeviceToHost);

	//Pooling���
	//*******************
	//**Fully Connected**
	//*******************
	/*
	//Weights ����
	float Weights[10][3][4][4];

	//Weights ����
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

	//GPU�� Weights��� ����
	float* dev_weights;
	cudaMalloc((void**)&dev_weights, sizeof(float) * 10 * 3 * 4 * 4);
	cudaMemcpy(dev_weights, Weights, sizeof(float) * 10 * 3 * 4 * 4, cudaMemcpyHostToDevice);
	//Weights�� ���� Filter ����ü ���� �� �Ҵ�
	cudnnFilterDescriptor_t weights_desc;
	cudnnCreateFilterDescriptor(&weights_desc);
	cudnnSetFilter4dDescriptor(weights_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 10, 3, 4, 4);
	//Fully Connected�� ���� Convolution ����ü ���� �� �Ҵ�
	cudnnConvolutionDescriptor_t fc_desc;
	cudnnCreateConvolutionDescriptor(&fc_desc);
	cudnnSetConvolution2dDescriptor(fc_desc, 0, 0, 1, 1, 1, 1, CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT);
	//Fully Connected ���� ��� ������� ����ü ����
	cudnnTensorDescriptor_t out_fc_desc;
	cudnnCreateTensorDescriptor(&out_fc_desc);
	//Fully Connected ������ ������ ����
	int out_fc_n;
	int out_fc_c;
	int out_fc_h;
	int out_fc_w;
	cudnnGetConvolution2dForwardOutputDim(fc_desc, out_fc_desc, weights_desc, &out_fc_n, &out_fc_c, &out_fc_h, &out_fc_w);
	//FC ������ ����

	//GPU�� FC ������ �Ҵ�
	float* dev_Output_FC;
	cudaMalloc((void**)&dev_Output_FC, sizeof(float) * ImageNum * 10 * 1 * 1);
	//FC ����ü �ʱ�ȭ
	cudnnSetTensor4dDescriptor(out_fc_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, ImageNum, 10, 1, 1);
	//FC ����ũ�� �Ҵ� �� ����
	size_t WS_size2 = 0;
	cudnnGetConvolutionForwardWorkspaceSize(cudnn, out_pool_desc, weights_desc, fc_desc, out_fc_desc, alg, &WS_size2);
	size_t* dev_WS2;
	cudaMalloc((void**)&dev_WS2, WS_size2);
	//Fully Connected ����

	//FC ����� CPU�� ����

	//*************************
	//**Fully Conncected Bias**
	//*************************

	const int ten = 10;
	/*
	//FC bias��
	float biasValueFC[10];

	for (int i = 0; i < ten; i++)
	{
	biasValueFC[i] = -5.0f;
	}
	*/

	std::vector<float> biasValueFC;
	biasValueFC.resize(10);
	InitWeightsbias(10, biasValueFC);

	//GPU�� FC bias�� ����
	float* dev_Bias_FC;
	cudaMalloc((void**)&dev_Bias_FC, sizeof(float) * 10);
	cudaMemcpy(dev_Bias_FC, &biasValueFC[0], sizeof(float) * 10, cudaMemcpyHostToDevice);

	cudnnTensorDescriptor_t out_Bias_FC_desc;
	cudnnCreateTensorDescriptor(&out_Bias_FC_desc);
	cudnnSetTensor4dDescriptor(out_Bias_FC_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, ten, 1, 1);
	//bias ���� ����

	//***********
	//**Softmax**
	//***********

	float* dev_Output_Softmax;
	cudaMalloc((void**)&dev_Output_Softmax, sizeof(float) * ImageNum * 10);

	//****************************
	//****************************
	//**Backpropagation �������**
	//****************************

	//***************************
	//**Softmax Backpropagation** - p - y
	//***************************

	//������� ����
	//float SoftBack[ImageNum][10][1][1];
	//GPU �޸� �Ҵ�
	float* dif_Soft_Back;
	cudaMalloc((void**)&dif_Soft_Back, sizeof(float) * ImageNum * 10);
	//����ü ���� �� �ʱ�ȭ
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
	//GPU �޸�
	float* dev_FC_bias_Back;
	cudaMalloc((void**)&dev_FC_bias_Back, sizeof(float));

	//**********************************
	//**Fc�� W delta Backpropagtion**
	//**********************************

	float* dev_Filter_Gradient;
	cudaMalloc((void**)&dev_Filter_Gradient, sizeof(float) * 3 * 10 * 4 * 4);
	// Workspace
	size_t WS_size3 = 0;
	cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn, weights_desc, dif_soft_back, fc_desc, out_pool_desc, CUDNN_CONVOLUTION_BWD_DATA_ALGO_0, &WS_size3);
	//GPU�� workspace �޸� �Ҵ�
	size_t* dev_WS3;
	cudaMalloc((void**)&dev_WS3, WS_size3);

	//**********************************
	//**fc �ٷ���, pooling ����� delta Backpropagtion**
	//**********************************

	float* dev_data_Gradient;
	cudaMalloc((void**)&dev_data_Gradient, sizeof(float) * ImageNum * 3 * 4 * 4);


	//***************************
	//**Pooling Backpropagation**
	//***************************

	float* dev_Pool_Back;
	cudaMalloc((void**)&dev_Pool_Back, sizeof(float) *ImageNum * 3 * 8 * 8);
	//����ü ������� �ʱ�ȭ
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

	//GPU �޸�
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
	//GPU�� workspace �޸� �Ҵ�
	size_t* dev_WS4;
	cudaMalloc((void**)&dev_WS4, WS_size4);


	float OutSoft[ImageNum][10][1][1];



	for (int iter = 0; iter < 10; ++iter)
	{

		//������
		cudnnConvolutionForward(cudnn, &alpha, in_desc, dev_Input, filt_desc, dev_Filt, conv_desc, alg, dev_WS, WS_size, &beta, out_conv_desc, dev_Output_Conv);


		cudaMemcpy(Output_Conv, dev_Output_Conv, sizeof(float) * out_conv_n* out_conv_c* out_conv_h* out_conv_w, cudaMemcpyDeviceToHost);

		std::cout << std::endl << std::endl << "Convolution ���" << std::endl << std::endl;
		for (int i = 0; i < out_conv_h; i++)
		{
			for (int j = 0; j < out_conv_w; j++)
			{
				std::cout << setw(10) << Output_Conv[0][0][i][j] << "  ";
			}
			std::cout << std::endl;
		}


		//bias ���


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


		std::cout << std::endl << std::endl << "Activation Function ���" << std::endl << std::endl;

		for (int i = 0; i < outputDimHW; i++)
		{
			for (int j = 0; j < outputDimHW; j++)
			{
				std::cout << setw(5) << Output_Activation[0][0][i][j] << "  ";
			}
			std::cout << std::endl;
		}


		// pooling ���
		cudnnPoolingForward(cudnn, pool_desc, &alpha, out_conv_desc, dev_Output_Act, &beta, out_pool_desc, dev_Output_Pool);
		float Output_Pool[ImageNum][FeatureNum][(outputDimHW + 2 * pool_pad_h) / pool_strd_h][(outputDimHW + 2 * pool_pad_w) / pool_strd_w];
		cudaMemcpy(Output_Pool, dev_Output_Pool,
			sizeof(float) * out_pool_n * out_pool_c * out_pool_h * out_pool_w, cudaMemcpyDeviceToHost);

		std::cout << std::endl << std::endl << "Pooling ���" << std::endl << std::endl;

		for (int i = 0; i < out_pool_h; i++)
		{
			for (int j = 0; j < out_pool_w; j++)
			{
				std::cout << setw(5) << Output_Pool[0][0][i][j] << "  ";
			}
			std::cout << std::endl;
		}


		//Fully Connected ����
		cudnnConvolutionForward(cudnn, &alpha, out_pool_desc, dev_Output_Pool, weights_desc, dev_weights, fc_desc, alg, dev_WS2, WS_size2, &beta, out_fc_desc, dev_Output_FC);

		float Output_FC[ImageNum][10][1][1];
		cudaMemcpy(Output_FC, dev_Output_FC, sizeof(float) * ImageNum * 10 * 1 * 1, cudaMemcpyDeviceToHost);

		std::cout << std::endl << std::endl << "Fully Connected  ���" << std::endl << std::endl;

		for (int n = 0; n < ImageNum; n++)
		{
			for (int c = 0; c < 10; c++) {

				cout << setw(10) << Output_FC[n][c][0][0] << "::";

			}cout << endl;

		}

		//FC bias ���� ����
		beta = 1.0f;
		cudnnAddTensor(cudnn, &alpha, out_Bias_FC_desc, dev_Bias_FC, &beta, out_fc_desc, dev_Output_FC);
		float Output_FC_Bias[ImageNum][10][1][1];
		cudaMemcpy(Output_FC_Bias, dev_Output_FC, sizeof(float) * ImageNum * ten, cudaMemcpyDeviceToHost);

		std::cout << std::endl << std::endl << "FC bias ���" << std::endl << std::endl;

		for (int n = 0; n < ImageNum; n++)
		{
			for (int c = 0; c < 10; c++) {

				cout << setw(10) << Output_FC_Bias[n][c][0][0] << "::";

			}cout << endl;

		}


		beta = 0;
		//softmax 


		cudnnSoftmaxForward(cudnn, CUDNN_SOFTMAX_FAST, CUDNN_SOFTMAX_MODE_INSTANCE, &alpha, out_fc_desc, dev_Output_FC, &beta, out_fc_desc, dev_Output_Softmax);

		std::cout << std::endl << std::endl << "softmax ���" << std::endl << std::endl;

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


		// cost function (ũ�ν� ��Ʈ����)���� ���� ��� 
		float sum2 = 0;
		float sum3 = 0;
		for (int n = 0; n < ImageNum; n++)
		{
			for (int c = 0; c < 10; c++) {

				sum3 = sum3 + (OutSoft[n][c][0][0] * TargetY[n][c]);

				sum2 = sum2 + (-log(OutSoft[n][c][0][0]) * TargetY[n][c]);
			}cout << endl;

			cout << "error(ũ�ν���Ʈ���� cost) :: " << sum2 << endl;
			cout << "��Ȯ�� :: " << sum3 << "  %" << endl;
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

			cout << suma[n] << "< = 1�� ������ ���ڴ�" << endl;
		}

		// ���� ó�� 
		/*

		float  *loss;
		cudaMalloc((void**)&loss, sizeof(float) * numImgs * 10 * 1 * 1);
		// ��, �󺧼�, �̹��� �Ѽ�, ����Ʈ�ƽ� ���
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



		cout << "�������������� Output_dloss_data" << endl;
		for (int n = 0; n < ImageNum; n++)
		{
			for (int c = 0; c < 10; c++) {

				cout << setw(10) << Output_loss[n][c] << "::";

			}cout << endl;

		}



		beta = 1.0f;
		//FC BACK bias delta ���
		cudnnConvolutionBackwardBias(cudnn, &alpha, out_fc_desc, loss, &beta, bias_desc, dev_FC_bias_Back);
		beta = 0;

		//FC Weight delta ���
		cudnnConvolutionBackwardFilter(cudnn, &alpha, out_pool_desc, dev_Output_Pool, dif_soft_back, loss, fc_desc, CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0, dev_WS3, WS_size3, &beta, weights_desc, dev_Filter_Gradient);

		//FC X Delta ���  
		cudnnConvolutionBackwardData(cudnn, &alpha, weights_desc, dev_weights, dif_soft_back, loss, fc_desc, CUDNN_CONVOLUTION_BWD_DATA_ALGO_0, dev_WS3, WS_size3, &beta, out_pool_desc, dev_data_Gradient);

		//Pooling delta���
		cudnnPoolingBackward(cudnn, pool_desc, &alpha, out_pool_desc, dev_Output_Pool, out_pool_desc, dev_data_Gradient, out_conv_desc, dev_Output_Act, &beta, pool_back_desc, dev_Pool_Back);

		//Activat delta���
		cudnnActivationBackward(cudnn, act_desc, &alpha, out_conv_desc, dev_Output_Act, out_conv_desc, dev_Pool_Back, out_conv_desc, dev_Output_Conv, &beta, out_conv_desc, dev_Act_Back);

		beta = 1.0f;
		//Bias delta ���
		cudnnConvolutionBackwardBias(cudnn, &alpha, out_conv_desc, dev_Act_Back, &beta, bias_desc, dev_Conv_bias_Back);
		beta = 0;
		//Weight delta ���
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
		cout << "no �� ���� �Ф�" << endl;
		//++num_errors;



		// cost function (ũ�ν� ��Ʈ����)���� ���� ���
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
				cout << "no �� ���� �Ф�" << endl;
			//++num_errors;

		}

	}



	std::cout << "��==========================================" << endl;

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