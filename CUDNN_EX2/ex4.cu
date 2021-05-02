#include <cuda.h>
#include <cudnn.h>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <string.h>
#include <iostream>

using namespace std;
using namespace cv;

// cu ���Ͽ��� ���� ��� �̹��� �ϳ� �ҷ����� 
int main()
{
    size_t imgCounter = 0;
    Mat img = imread("D:\\DataSet\\cifar\\test\\0_cat.png");	// �̹��������� �о� �鿩 Mat �������� �����Ű��
    unsigned char* imgd = img.data;
    size_t imgEleSize = img.elemSize();		// ���ȼ��� ����������, rgb = 3
    size_t imgWidth = img.cols;				// �� ��, �̹��� ���� ũ��, �Է� �������� ���� ����
    size_t imgHeight = img.rows;			// ���� ��, �̹��� ���� ũ��, �Է� �������� ���� ����
    size_t imgChannel = img.channels();		// ����, �̹��� ä�� ��, �Է� �������� ä�� ��
    vector<vector<vector<vector<int>>>> InData_NCHW(10000);

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
	cout << "imgWidth :: " << imgWidth  << endl;
	cout << "imgHeight :: " << imgHeight << endl;
	cout << "imgChannel :: " << imgChannel << endl;
    //  �ҷ��� �̹��� ������ �� Ȯ��
    cout << imgCounter << " ��° �̹��� ���� :: ";
	cout << endl;
	for (int c = 0; c < imgChannel; c++)
	{
	for (int y = 0; y < imgHeight; y++)
	{
		for (int x = 0; x < imgWidth; x++)
		{
			cout << setw(3)<< InData_NCHW[imgCounter][c][y][x]<<
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





    return 0;
}