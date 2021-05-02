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


	 //�Էº���
	const int ImageNum = 1;
	const int FeatureNum = 3;
	const int FeatureHeight = 8;
	const int FeatureWidth = 8;

	// 4�� ��� ���� �Ҵ� ����.
	float Input[ImageNum][FeatureNum][FeatureHeight][FeatureWidth];

	int count = 1;
	// mat ���� - > 4�� ���
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
	
	// input 4�� �迭 ���� ������ NCHW 1,3,8,8
	cout << "input 4�� �迭" << endl;

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

	//GPU�� �Է���� �޸� �Ҵ� �� �� ����
	float * dev_Input;
	cudaMalloc((void**)&dev_Input, sizeof(Input) );
	cudaMemcpy(dev_Input, Input, sizeof(Input), cudaMemcpyHostToDevice);

	//�Է���� ����ü ����, �Ҵ�, �ʱ�ȭ
	cudnnTensorDescriptor_t in_desc; //�Է� ������ �� ������ ���� �ִ� ����ü�� ����Ű�� ���� ������
	cudnnCreateTensorDescriptor(&in_desc); // 4D tensor ����ü ��ü ����
	cudnnSetTensor4dDescriptor( // 4D tensor ����ü �ʱ�ȭ �Լ�
		/*tensorDesc,*/ in_desc,
		/*format,*/CUDNN_TENSOR_NCHW,
		/*dataType,*/CUDNN_DATA_FLOAT,
		/*Number of images*/ImageNum,
		/*C*/FeatureNum,
		/*H*/FeatureHeight,
		/*W*/FeatureWidth);

	//���� ������ ���� - ���� ���� �����ϵ��� KCRS

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


	//Filter ����
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

	cout << "Filter���" << endl;

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

	//GPU�� ������� ����
	float * dev_Filt;
	cudaMalloc((void**)&dev_Filt, sizeof(float) * filt_n * filt_c * filt_h * filt_w);
	cudaMemcpy(dev_Filt, Filter, sizeof(float) * filt_n * filt_c * filt_h * filt_w, cudaMemcpyHostToDevice);

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
	const int pad_h = 1; //padding ����
	const int pad_w = 1; //padding ����
	const int str_h = 1; //stride ����
	const int str_w = 1; //stride ����
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
	const int outputDimHW = 4;

	//Convolution��� ����
	float Output_Conv[ImageNum][FeatureNum][outputDimHW][outputDimHW];


	//GPU�� Convolution ��� ��� �Ҵ�
	float * dev_Output_Conv;
	cudaMalloc((void**)&dev_Output_Conv, sizeof(float) * out_conv_c * out_conv_h * out_conv_n * out_conv_w);

	//Convolution ����ü �ʱ�ȭ
	cudnnSetTensor4dDescriptor(out_conv_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
		ImageNum, FeatureNum, outputDim, outputDim);

	//�Է°� ����, ������� �е�, ��Ʈ���̵尡 ���� ���� �־������� ���� ���� �˰����� ���������� �˾Ƴ���
	cudnnConvolutionFwdAlgo_t alg;
	alg = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;

	//Conv ���� ������ũ�� �˾Ƴ��� �� �� ���� �޸� �Ҵ� �߰�
	size_t WS_size = 0;
	cudnnGetConvolutionForwardWorkspaceSize(// This function returns the amount of GPU memory workspace
		cudnn, in_desc, filt_desc, conv_desc, out_conv_desc, alg, &WS_size);

	size_t * dev_WS;
	cudaMalloc((void**)&dev_WS, WS_size);

	//����
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

	//Convolution��� GPU�� ����
	cudaMemcpy(Output_Conv, dev_Output_Conv, sizeof(float) * 16*3, cudaMemcpyDeviceToHost);

	cout << "Convolution���" << endl;
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

	

		//Bias ��� ������� ����
		float Output_Bias[ImageNum][FeatureNum][outputDimHW][outputDimHW];

		//bias �� ����

		float biasValue[filt_n];

		for (int i = 0; i < filt_n; i++) {
			 biasValue[i]= 0.0f;
		}
		for (int i = 0; i < filt_n; i++) {
			cout << biasValue[i] << "  ";
		}

		//GPU�� bias�� ����
		float * dev_Bias;
		cudaMalloc((void**)&dev_Bias, sizeof(biasValue));
		cudaMemcpy(dev_Bias, biasValue, sizeof(biasValue), cudaMemcpyHostToDevice);

		//bias��� ������� ����, �Ҵ�
		cudnnTensorDescriptor_t bias_desc;
		cudnnCreateTensorDescriptor(&bias_desc);
		cudnnSetTensor4dDescriptor( bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, filt_n, 1, 1);

		//bias ���� ���� 
		cudnnAddTensor(cudnn, &alpha, bias_desc, dev_Bias,
			&alpha, /*input -> output*/out_conv_desc, /*input -> output*/dev_Output_Conv);

		//Bias�� ���
		cudaMemcpy(Output_Bias, dev_Output_Conv,
			sizeof(float) * ImageNum * FeatureNum * outputDimHW * outputDimHW, cudaMemcpyDeviceToHost);



		//Bias�� ���
	
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



		//Activation Function ����ü ���� �� �Ҵ� 
		cudnnActivationDescriptor_t act_desc;
		cudnnCreateActivationDescriptor(&act_desc);

		//Activation Function ���� ���� - ���� ���������ϵ���
		cudnnActivationMode_t Activation_Function;
		Activation_Function = CUDNN_ACTIVATION_RELU;
		//Activation_Function = CUDNN_ACTIVATION_TANH; 
		//Activation_Function = CUDNN_ACTIVATION_SIGMOID;

		cudnnSetActivationDescriptor(act_desc, Activation_Function, CUDNN_PROPAGATE_NAN, 0);

		//Activation Function �޸� GPU�� ����
		float * dev_Output_Act;
		cudaMalloc((void**)&dev_Output_Act, sizeof(float) * outputDimHW * outputDimHW * 3);


		//Activatin Function �������
		cudnnActivationForward(
			cudnn, act_desc, &alpha, out_conv_desc, dev_Output_Conv,
			&beta, out_conv_desc, dev_Output_Act);

		//Activation Function ����� ���� ���
		float Output_Activation[ImageNum][FeatureNum][outputDimHW][outputDimHW];
		cudaMemcpy(Output_Activation, dev_Output_Act, sizeof(float) * ImageNum * FeatureNum * outputDimHW * outputDimHW, cudaMemcpyDeviceToHost);

		//Actavation Function ���
		
		std::cout << std::endl << std::endl << "Activation Function ���" << std::endl << std::endl;
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



		//**Pooling����**


		//Pooling ���꿡�� ���� ���� - ���� ���� �����ϵ���

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
		cudnnGetPooling2dForwardOutputDim(pool_desc, out_conv_desc,
			&out_pool_n, &out_pool_c, &out_pool_h, &out_pool_w);

		//GPU�� Pooling ������ �޸��Ҵ�
		float * dev_Output_Pool;
		cudaMalloc((void**)&dev_Output_Pool,
			sizeof(float) * out_pool_n * out_pool_c * out_pool_h * out_pool_w);

		//Pooling ������� ����ü �ʱ�ȭ
		cudnnSetTensor4dDescriptor(out_pool_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
			out_pool_n, out_pool_c, out_pool_h, out_pool_w);

		//Pooling���� ����
		cudnnPoolingForward(cudnn, pool_desc, &alpha, out_conv_desc, dev_Output_Act,
			&beta, out_pool_desc, dev_Output_Pool);

		//Pooling���
		float Output_Pool[ImageNum][FeatureNum][(outputDimHW + 2 * pool_pad_h) / pool_strd_h][(outputDimHW + 2 * pool_pad_w) / pool_strd_w];
		cudaMemcpy(Output_Pool, dev_Output_Pool,
			sizeof(float) * out_pool_n * out_pool_c * out_pool_h * out_pool_w, cudaMemcpyDeviceToHost);

		//Pooling��� 
		
		std::cout << std::endl << std::endl << "Pooling ���" << std::endl << std::endl;

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

		//Weights ����
		float Weights[10][3][2][2];

		//Weights ����
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



		//GPU�� Weights��� ����
		float * dev_weights;
		cudaMalloc((void**)&dev_weights, sizeof(float) * 10 * 3 * 2 * 2);
		cudaMemcpy(dev_weights, Weights, sizeof(float) * 10 * 3 * 2 * 2, cudaMemcpyHostToDevice);

		//Weights�� ���� Filter ����ü ���� �� �Ҵ�
		cudnnFilterDescriptor_t weights_desc;
		cudnnCreateFilterDescriptor(&weights_desc);
		cudnnSetFilter4dDescriptor(weights_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 10, 3, 2, 2);

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

		cudnnGetConvolution2dForwardOutputDim(
			fc_desc, out_fc_desc, weights_desc, &out_fc_n, &out_fc_c, &out_fc_h, &out_fc_w);

		//FC ������ ����
		float Output_FC[ImageNum][10][1][1];

		//GPU�� FC ������ �Ҵ�
		float *dev_Output_FC;
		cudaMalloc((void**)&dev_Output_FC, sizeof(float) * ImageNum * 10 * 1 * 1);

		//FC ����ü �ʱ�ȭ
		cudnnSetTensor4dDescriptor(out_fc_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, ImageNum, 10, 1, 1);


		//FC ����ũ�� �Ҵ� �� ����
		size_t WS_size2 = 0;
		cudnnGetConvolutionForwardWorkspaceSize(
			cudnn, out_pool_desc, weights_desc, fc_desc, out_fc_desc, alg, &WS_size2);

		size_t * dev_WS2;
		cudaMalloc((void**)&dev_WS2, WS_size2);

		//Fully Connected ���� 
		cudnnConvolutionForward(
			cudnn, &alpha, out_pool_desc, dev_Output_Pool, weights_desc, dev_weights, fc_desc,
			alg, dev_WS2, WS_size2, &beta, out_fc_desc, dev_Output_FC);

		//FC ����� CPU�� ����
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
		

		//FC bias ��� ����
		float Output_FC_Bias[ImageNum][10][1][1];

		//FC bias��
		float biasValueFC[10];


		for (int i = 0; i < 10; i++) {
			biasValueFC[i] = -5.0f;
		}
		for (int i = 0; i < 10; i++) {
			cout << biasValueFC[i] << "  ";
		}



		//GPU�� FC bias�� ����
		float * dev_Bias_FC;
		cudaMalloc((void**)&dev_Bias_FC, sizeof(float)*10);
		cudaMemcpy(dev_Bias_FC, biasValueFC, sizeof(float)* 10, cudaMemcpyHostToDevice);


		//FC Softmax ����ü - �� �� ��������?
		cudnnTensorDescriptor_t out_Bias_FC_desc;
		cudnnCreateTensorDescriptor(&out_Bias_FC_desc);
		cudnnSetTensor4dDescriptor(out_Bias_FC_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 10, 1, 1);


		//bias ���� ����
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


		cudnnSoftmaxForward(cudnn, CUDNN_SOFTMAX_FAST, CUDNN_SOFTMAX_MODE_INSTANCE/*�� �κ� �ſ� �߿� - INSTANCE�� �ؾ� �ٷ� �̺а��*/,
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



		 //�������
		 float FCbiasBack[ImageNum][10][1][1];

		 //GPU �޸�
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

		 //������� ����
		 float FC_delta_Weights[10][3][2][2];

		 //GPU�� �޸� �Ҵ�
		 float * dev_Filter_Gradient;
		 cudaMalloc((void**)&dev_Filter_Gradient, sizeof(float) * 3 * 10 * 2 * 2);

		 // Workspace
		 size_t WS_size3 = 0;
		 cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn, weights_desc, out_fc_desc, fc_desc, out_pool_desc, CUDNN_CONVOLUTION_BWD_DATA_ALGO_0, &WS_size3);

		 //GPU�� workspace �޸� �Ҵ�
		 size_t * dev_WS3;
		 cudaMalloc((void**)&dev_WS3, WS_size3);

		 //Fully Connected Backpropagation delta
		 cudnnConvolutionBackwardFilter(cudnn, &alpha,
			 out_pool_desc, dev_Output_Pool, out_fc_desc, dev_dloss, fc_desc, CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,
			 dev_WS3, WS_size3, &beta, weights_desc, dev_Filter_Gradient);

		 //CPU�� ��� ����
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
		 //**fc �ٷ���, pooling ����� delta Backpropagtion**
		 //**********************************


		 //������� ����
		 float FC_delta_data[ImageNum][3][2][2]; // fc �ٷ� �� �� 

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
		 //CPU�� ��� ����
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

		 //Pooling Backward ������� ���� �� GPU�� �޸� �Ҵ�
		 float PoolingBack[ImageNum][3][4][4];


		 float * dev_Pool_Back;
		 cudaMalloc((void**)&dev_Pool_Back, sizeof(float) *ImageNum * 3 * 4 * 4);


		 //����ü ������� �ʱ�ȭ
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

		 //�������
		 float ConvbiasBack[3];

		 //GPU �޸�
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

		 //GPU�� workspace �޸� �Ҵ�
		 size_t * dev_WS4;
		 cudaMalloc((void**)&dev_WS4, WS_size4);

		 //Fully Connected Backpropagation delta
		 cudnnConvolutionBackwardFilter(cudnn, &alpha,
			 in_desc, dev_Input, //  x �Է°�				  // ImageNum, 3, 32, 32
			 out_conv_desc, dev_Act_Back, // dy �ٷ��� act_dev // ImageNum, 3, 8, 8
			 conv_desc, CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,
			 dev_WS4, WS_size4, &beta,
			 filt_desc, dev_Filter_conv2); // dw ����			// 3 ,3, 8, 8

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