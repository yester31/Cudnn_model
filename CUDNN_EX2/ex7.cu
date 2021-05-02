#include <cuda.h>
#include <cudnn.h>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <string.h>
#include <iostream>

using namespace std;
using namespace cv;

// cu 파일에서 절대 경로 이미지 하나 불러오기 
int main()
{
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
				Input[0][i][y][x] = imgd[3 * 32 * y + 3 * x + i];
				
			}
		}
	}


	//  불러온 이미지 데이터 값 확인
	cout << ImageNum << " 번째 이미지 파일 :: ";
	cout << endl;
	for (int c = 0; c < 3; c++)
	{
		for (int y = 0; y < 32; y++)
		{
			for (int x = 0; x < 32; x++)
			{
				cout << setw(3) << Input[0][c][y][x] <<
					" "; // 픽셀에 저장된 데이터 값 (첫번째 행 출력)
			}
			cout << endl;
		}
		cout << endl;// BGR 순서 
		cout << "===================================================" << endl;
	}
	cout << endl;
	cout << "===================================================" << endl;
	//imgCounter += 1;





	return 0;
}