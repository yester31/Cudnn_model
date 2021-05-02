#include <cuda.h>
#include <cudnn.h>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <string.h>
#include <iostream>

using namespace std;
using namespace cv;

void checkCUDNN(cudnnStatus_t status)

{
    if (status != CUDNN_STATUS_SUCCESS)
        cout << "[ERROR] CUDNN " << status << endl;
}

void checkCUDA(cudaError_t error)

{
    if (error != CUDA_SUCCESS)
        cout << "[ERROR] CUDA " << error << endl;
}

void print(char* title, float* src, int filter_num, int h, int w)

{
    cout << title << endl;

    for (int i = 0; i < filter_num; i++)
    {
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                printf("%.0f ", src[i * h * w + y * w + x]);
            }

            cout << endl;
        }

        cout << endl;
    }
}

int main()
{
    size_t imgCounter = 1;
    Mat img = imread("D:\\DataSet\\cifar\\test\\0_cat.png");	// �̹��������� �о� �鿩 Mat �������� �����Ű��
    unsigned char* imgd = img.data;
    size_t imgEleSize = img.elemSize();		// ���ȼ��� ����������, rgb = 3
    size_t imgWidth =
        img.cols;				// �� ��, �̹��� ���� ũ��, �Է� �������� ���� ����
    size_t imgHeight =
        img.rows;			// ���� ��, �̹��� ���� ũ��, �Է� �������� ���� ����
    size_t imgChannel = img.channels();		// ����, �̹��� ä�� ��, �Է� �������� ä�� ��
    vector<vector<vector<vector<int>>>> InData_NCHW(10);

    //�Է� ������ ���� N C H W

    for (int i = 0; i < imgChannel; i++)
    {
        InData_NCHW[imgCounter].resize(imgChannel);

        for (int y = 0; y < imgHeight; y++)
        {
            InData_NCHW[imgCounter][i].resize(imgHeight);

            for (int x = 0; x < imgWidth; x++)
            {
                InData_NCHW[imgCounter][i][y].resize(imgWidth);
                InData_NCHW[imgCounter][i][y][x] = imgd[imgChannel * imgWidth  * y + imgChannel * x + i];
            }
        }
    }

    //  �ҷ��� �̹��� ������ �� Ȯ��
    cout << imgCounter << " ��° �̹��� ���� :: ";
    cout << endl;

    for (int c = 0; c < imgChannel; c++)
    {
        for (int y = 0; y < imgHeight; y++)
        {
            for (int x = 0; x < imgWidth; x++)
            {
                cout << setw(3) << InData_NCHW[imgCounter][c][y][x] <<
                     " "; // �ȼ��� ����� ������ �� (ù��° �� ���)
            }

            cout << endl;
        }

        cout << endl;// BGR ����
        cout << "===================================================" << endl;
    }

    cout << endl;
    cout << "===================================================" << endl;
    //imgCounter += 1;
    const int batch_count = imgCounter;//�Է� ������ ����, ��ġ������
    const int in_channel = imgChannel;//�Է� �������� ä�� ��
    const int in_height = imgHeight;//�Է� �������� ���� ����
    const int in_width = imgWidth;//�Է� �������� ���� ����
    const int out_channel = 2;//��� Ŭ���� ��
    const int filter_width = 3;//������� ����(����ġ)�� ���� ����
    const int filter_height = 3;//������� ����(����ġ)�� ���� ����
    const int filter_num = 1;//������� ����(����ġ) ����
    const int padding_w = 1;//������� �е�. ������ ���� ���� ���̰� 3�̰� �е��� 1,1 �̸� SAME Convolution�� �ȴ�
    const int padding_h = 1;
    const int stride_horizontal = 1;//������� ��Ʈ���̵�
    const int stride_vertical = 1;
    const int pool_window_w = 2;
    const int pool_window_h = 2;
    const int pool_stride_horizontal = 2;
    const int pool_stride_vertical = 2;
    const int pool_padding_horizontal = 0;
    const int pool_padding_vertical = 0;
    const int pool_w = in_width / pool_stride_horizontal;
    const int pool_h = in_height / pool_stride_vertical;
    const int src_len = batch_count * filter_num * in_height * in_width;
    const int pool_len = batch_count * filter_num * pool_w * pool_h;

  //host ��� ������
	vector<vector<vector<vector<int>>>> outData(10);

	for (int i = 0; i < filter_num; i++)
	{
		outData[imgCounter].resize(filter_num);

		for (int y = 0; y < in_height; y++)
		{
			outData[imgCounter][i].resize(in_height);

			for (int x = 0; x < in_width; x++)
			{
				outData[imgCounter][i][y].resize(in_width);
			}
		}
	}


    float* hostArray = new float[src_len];
    float* inData_d;//device �Է� ������
    float* outData_d, *outData1_d;//device ��� ������
    float* filterData_d;//device ������� ���� ������
    float* filterData2_d;//device FCN ���� ������
    float* biasData_d;
    void* workSpace;//CUDNN�� �۾� �߿� ����� ���� �޸�



    //����(����ġ) ����
    float filterData[filter_num][2][filter_height][filter_width] =
    {

        {

            { { 0.0f, 0.0f, 0.0f }, { 0.0f, 1.0f, 0.0f }, { 0.0f, 0.0f, 0.0f } },

            { { 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 1.0f }, { 0.0f, 0.0f, 0.0f } }

        }

    };
    //Fully connected Layer ����ġ


	vector<vector<vector<vector<float>>>> filterData2;
	for (int z = 0; z < out_channel; z++)
	{
		outData.resize(out_channel);
	for (int i = 0; i < filter_num; i++)
	{
		outData[imgCounter].resize(filter_num);

		for (int y = 0; y < pool_h; y++)
		{
			outData[imgCounter][i].resize(pool_h);

			for (int x = 0; x < pool_w; x++)
			{
				outData[imgCounter][i][y].resize(pool_w);
			}
		}
	}
	}
    filterData2 =
    {

        { { { 0.1f, 0.1f }, { 0.1f, 0.1f } } },

        { { { 0.2f, 0.2f }, { 0.2f, 0.2f } } }

    };
    float biasData[filter_num] =
    {

        -20

    };
    cout << "in_NCHW" << endl;

    for (int i = 0; i < in_channel; i++)
    {
        for (int y = 0; y < in_height; y++)
        {
            for (int x = 0; x < in_width; x++)
            {
                printf("%.0f ", InData_NCHW[1][i][y][x]);
            }

            cout << endl;
        }

        cout << endl;
    }

    cout << "weights" << endl;

    for (int n = 0; n < filter_num; n++)
    {
        for (int i = 0; i < in_channel; i++)
        {
            for (int y = 0; y < filter_height; y++)
            {
                for (int x = 0; x < filter_width; x++)
                {
                    printf("%.1f ", filterData[n][i][y][x]);
                }

                cout << endl;
            }

            cout << endl;
        }
    }

    //GPU �޸� �Ҵ�
    checkCUDA(cudaMalloc((void**)&inData_d, sizeof(InData_NCHW)));
    checkCUDA(cudaMalloc((void**)&outData_d, sizeof(outData)));
    checkCUDA(cudaMalloc((void**)&outData1_d, sizeof(outData)));
    checkCUDA(cudaMalloc((void**)&filterData_d, sizeof(filterData)));
    checkCUDA(cudaMalloc((void**)&biasData_d, sizeof(biasData)));
    checkCUDA(cudaMalloc((void**)&filterData2_d, sizeof(filterData2)));
    //CPU �����͸� GPU �޸𸮷� ����
    //NHWC �� NCHW �߿� �����մϴ�.
    checkCUDA(cudaMemcpy(inData_d, InData_NCHW, sizeof(InData_NCHW), cudaMemcpyHostToDevice));

    checkCUDA(cudaMemcpy(filterData_d, filterData, sizeof(filterData), cudaMemcpyHostToDevice));
    checkCUDA(cudaMemcpy(filterData2_d, filterData2, sizeof(filterData2), cudaMemcpyHostToDevice));
    checkCUDA(cudaMemcpy(biasData_d, biasData, sizeof(biasData), cudaMemcpyHostToDevice));
    //CUDNN �迭
    cudnnHandle_t cudnnHandle;// CUDNN�� ����ϱ� ���� �ڵ鷯
    cudnnTensorDescriptor_t inTensorDesc, outTensorDesc, biasTensorDesc, poolOutTensorDesc,
                            sftTensorDesc;//������ ����ü ����
    cudnnFilterDescriptor_t filterDesc, filterDesc2;//���� ����ü ����
    cudnnConvolutionDescriptor_t convDesc;//������� ����ü ����
    cudnnConvolutionDescriptor_t convDesc2;//������� ����ü ����
    cudnnPoolingDescriptor_t poolDesc;//Ǯ�� ����ü ����
    cudnnActivationDescriptor_t actDesc;//Ȱ���Լ� ����ü ����
    //�Ҵ�
    checkCUDNN(cudnnCreate(&cudnnHandle));
    checkCUDNN(cudnnCreateTensorDescriptor(&inTensorDesc));
    checkCUDNN(cudnnCreateTensorDescriptor(&outTensorDesc));
    checkCUDNN(cudnnCreateTensorDescriptor(&biasTensorDesc));
    checkCUDNN(cudnnCreateTensorDescriptor(&poolOutTensorDesc));
    checkCUDNN(cudnnCreateTensorDescriptor(&sftTensorDesc));
    checkCUDNN(cudnnCreateFilterDescriptor(&filterDesc));
    checkCUDNN(cudnnCreateFilterDescriptor(&filterDesc2));
    checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));
    checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc2));
    checkCUDNN(cudnnCreatePoolingDescriptor(&poolDesc));
    checkCUDNN(cudnnCreateActivationDescriptor(&actDesc));
    //�ʱ�ȭ
    //inData_NCHW ���� - ������ [Number][Channel][Height][Width] �������� �˷���
    //checkCUDNN(cudnnSetTensor4dDescriptor(inTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_count, in_channel, in_height, in_width));
    //inData_NHWC ���� - ������ [Number][Height][Width][Channel] �������� �˷���
    checkCUDNN(cudnnSetTensor4dDescriptor(inTensorDesc,
                                          CUDNN_TENSOR_NHWC,
                                          CUDNN_DATA_FLOAT,
                                          batch_count,
                                          in_channel,
                                          in_height,
                                          in_width));
    checkCUDNN(cudnnSetFilter4dDescriptor(filterDesc,
                                          CUDNN_DATA_FLOAT,
                                          CUDNN_TENSOR_NCHW,
                                          filter_num,
                                          in_channel,
                                          filter_height,
                                          filter_width));
    checkCUDNN(cudnnSetFilter4dDescriptor(filterDesc2,
                                          CUDNN_DATA_FLOAT,
                                          CUDNN_TENSOR_NCHW,
                                          out_channel,
                                          filter_num,
                                          pool_h,
                                          pool_w));
    //��������� �е�, ��Ʈ���̵�, ������� ��� ���� ����
    checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc, padding_h, padding_w, stride_vertical,
                                               stride_horizontal, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
    //Ǯ�� Ŀ��Ƽ�� ��Ʈ��ũ �¾�
    checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc2, 0, 0, 2, 2, 1, 1, CUDNN_CROSS_CORRELATION,
                                               CUDNN_DATA_FLOAT));
    //���̾ �¾�
    checkCUDNN(cudnnSetTensor4dDescriptor(biasTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1,
                                          filter_num, 1, 1));
    //Ǯ�� �¾�
    checkCUDNN(cudnnSetPooling2dDescriptor(poolDesc, CUDNN_POOLING_MAX, CUDNN_PROPAGATE_NAN,
                                           pool_window_h, pool_window_w, pool_padding_vertical, pool_padding_horizontal, pool_stride_vertical,
                                           pool_stride_horizontal));
    //Ȱ���Լ� Relu �¾�
    checkCUDNN(cudnnSetActivationDescriptor(actDesc, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0));
    int out_n, out_c, out_h, out_w;
    //�Էµ����͸� ������ ������ ��� ������� ������ ��� �������� ���� �˾Ƴ���
    checkCUDNN(cudnnGetConvolution2dForwardOutputDim(convDesc, inTensorDesc, filterDesc, &out_n, &out_c,
                                                     &out_h, &out_w));
    printf("conv out shape (n x c x h x w) = (%d x %d x %d x %d)\n", out_n, out_c, out_h, out_w);
    checkCUDNN(cudnnSetTensor4dDescriptor(outTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, out_n,
                                          out_c, out_h, out_w));
    //Ǯ�� ��� ���� Ȯ��
    checkCUDNN(cudnnGetPooling2dForwardOutputDim(poolDesc, outTensorDesc, &out_n, &out_c, &out_h,
                                                 &out_w));
    printf("pool out shape (n x c x h x w) = (%d x %d x %d x %d)\n", out_n, out_c, out_h, out_w);
    //Ǯ�� ��� ���� �¾�
    checkCUDNN(cudnnSetTensor4dDescriptor(poolOutTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, out_n,
                                          out_c, out_h, out_w));
    //FCN ��� ���� Ȯ��
    checkCUDNN(cudnnGetConvolution2dForwardOutputDim(convDesc2, poolOutTensorDesc, filterDesc2, &out_n,
                                                     &out_c, &out_h, &out_w));
    printf("conv2 out shape (n x c x h x w) = (%d x %d x %d x %d)\n", out_n, out_c, out_h, out_w);
    checkCUDNN(cudnnSetTensor4dDescriptor(sftTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, out_n,
                                          out_c, out_h, out_w));
    //�Է°� ����, ������� �е�, ��Ʈ���̵尡 ���� ���� �־������� ���� ���� �˰����� ���������� �˾Ƴ���
    cudnnConvolutionFwdAlgo_t algo;
    checkCUDNN(cudnnGetConvolutionForwardAlgorithm(cudnnHandle,
                                                   inTensorDesc,
                                                   filterDesc,
                                                   convDesc,
                                                   outTensorDesc,
                                                   CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                                   0,
                                                   &algo
                                                  ));
    cout << "Fastest algorithm for conv0 = " << algo << endl;
    cudnnConvolutionFwdAlgo_t algo2;
    checkCUDNN(cudnnGetConvolutionForwardAlgorithm(cudnnHandle,
                                                   poolOutTensorDesc,
                                                   filterDesc2,
                                                   convDesc2,
                                                   sftTensorDesc,
                                                   CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                                   0,
                                                   &algo2
                                                  ));
    cout << "Fastest algorithm for conv1 = " << algo2 << endl;
    //������ �˾Ƴ� ���� ���� �˰����� ����� ��� ���������� �ʿ��� ���� �������� ũ�⸦ �˾Ƴ���
    size_t sizeInBytes = 0;
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle,
                                                       inTensorDesc,
                                                       filterDesc,
                                                       convDesc,
                                                       outTensorDesc,
                                                       algo,
                                                       &sizeInBytes));
    cout << "sizeInBytes " << sizeInBytes << endl;

    //���������� ���� �����Ͱ� �ʿ��� ��찡 �ִٸ� �޸� �Ҵ�

    if (sizeInBytes != 0) checkCUDA(cudaMalloc(&workSpace, sizeInBytes));

    float alpha = 1.0f;
    float beta = 0.0f;
    //������� ����
    //alpha�� beta�� "output = alpha * Op(inputs) + beta * output" �� ����
    //�Ϲ� ��������� output =   1   *    inputs
    //�׷���          output =   1   * Op(inputs) +   0  * output �� �ǵ��� alpha�� beta�� 1,0���� ������
    checkCUDNN(cudnnConvolutionForward(cudnnHandle,
                                       &alpha,
                                       inTensorDesc,
                                       inData_d,
                                       filterDesc,
                                       filterData_d,
                                       convDesc,
                                       algo,
                                       workSpace,
                                       sizeInBytes,
                                       &beta,
                                       outTensorDesc,
                                       outData_d));
    checkCUDA(cudaMemcpy(hostArray, outData_d, sizeof(float)* src_len, cudaMemcpyDeviceToHost));
    print("conv out", hostArray, filter_num, in_height, in_width);
    //Add Bias
    beta = 1.0f;
    checkCUDNN(cudnnAddTensor(cudnnHandle, &alpha, biasTensorDesc, biasData_d, &beta, outTensorDesc,
                              outData_d));
    checkCUDA(cudaMemcpy(hostArray, outData_d, sizeof(float)* src_len, cudaMemcpyDeviceToHost));
    print("Add bias out", hostArray, filter_num, in_height, in_width);
    //Activation - Relu
    beta = 0.0f;
    checkCUDNN(cudnnActivationForward(cudnnHandle, actDesc, &alpha, outTensorDesc, outData_d, &beta,
                                      outTensorDesc, outData1_d));
    checkCUDA(cudaMemcpy(hostArray, outData1_d, sizeof(float)* src_len, cudaMemcpyDeviceToHost));
    print("Activation - Relu out", hostArray, filter_num, in_height, in_width);
    //Pooling
    checkCUDNN(cudnnPoolingForward(cudnnHandle, poolDesc, &alpha, outTensorDesc, outData1_d, &beta,
                                   poolOutTensorDesc, outData_d));
    checkCUDA(cudaMemcpy(hostArray, outData_d, sizeof(float)* pool_len, cudaMemcpyDeviceToHost));
    print("pool out", hostArray, filter_num, pool_h, pool_w);
    //FC
    checkCUDNN(cudnnConvolutionForward(cudnnHandle,
                                       &alpha,
                                       poolOutTensorDesc,
                                       outData_d,
                                       filterDesc2,
                                       filterData2_d,
                                       convDesc2,
                                       algo2,
                                       workSpace,
                                       sizeInBytes,
                                       &beta,
                                       sftTensorDesc,
                                       outData1_d));
    checkCUDA(cudaMemcpy(hostArray, outData1_d, sizeof(float)* out_channel, cudaMemcpyDeviceToHost));
    print("FCN out", hostArray, out_channel, 1, 1);
    //Softmax
    cudnnSoftmaxForward(cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, &alpha,
                        sftTensorDesc, outData1_d, &beta, sftTensorDesc, outData_d);
    checkCUDA(cudaMemcpy(hostArray, outData_d, sizeof(float)* out_channel, cudaMemcpyDeviceToHost));
    print("Softmax out", hostArray, out_channel, 1, 1);
    //�޸� ����
    checkCUDNN(cudnnDestroyTensorDescriptor(inTensorDesc));
    checkCUDNN(cudnnDestroyTensorDescriptor(outTensorDesc));
    checkCUDNN(cudnnDestroyTensorDescriptor(biasTensorDesc));
    checkCUDNN(cudnnDestroyTensorDescriptor(poolOutTensorDesc));
    checkCUDNN(cudnnDestroyTensorDescriptor(sftTensorDesc));
    checkCUDNN(cudnnDestroyConvolutionDescriptor(convDesc));
    checkCUDNN(cudnnDestroyConvolutionDescriptor(convDesc2));
    checkCUDNN(cudnnDestroyPoolingDescriptor(poolDesc));
    checkCUDNN(cudnnDestroyFilterDescriptor(filterDesc));
    checkCUDNN(cudnnDestroyFilterDescriptor(filterDesc2));
    checkCUDNN(cudnnDestroyActivationDescriptor(actDesc));
    checkCUDNN(cudnnDestroy(cudnnHandle));
    checkCUDA(cudaFree(inData_d));
    checkCUDA(cudaFree(outData_d));;
    checkCUDA(cudaFree(filterData_d));
    checkCUDA(cudaThreadSynchronize());
    return 0;
}
