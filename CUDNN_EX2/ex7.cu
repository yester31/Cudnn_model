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
	//�Էº���
	const int ImageNum = 1;
	const int FeatureNum = 3;
	const int FeatureHeight = 32;
	const int FeatureWidth = 32;

	//********
	//**�Է�**
	//********
	Mat img = imread("D:\\DataSet\\cifar\\test\\0_cat.png");	// �̹��������� �о� �鿩 Mat �������� �����Ű��
	unsigned char* imgd = img.data;
	//�Է���� ����
	float Input[ImageNum][FeatureNum][FeatureHeight][FeatureWidth];

	//�Է���� ����
	for (int i = 0; i < 3; i++)
	{
		for (int y = 0; y < 32; y++)
		{
			for (int x = 0; x < 32; x++)
			{
				Input[0][i][y][x] = imgd[3 * 32 * y + 3 * x + i];
				
			}
		}
	}


	//  �ҷ��� �̹��� ������ �� Ȯ��
	cout << ImageNum << " ��° �̹��� ���� :: ";
	cout << endl;
	for (int c = 0; c < 3; c++)
	{
		for (int y = 0; y < 32; y++)
		{
			for (int x = 0; x < 32; x++)
			{
				cout << setw(3) << Input[0][c][y][x] <<
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