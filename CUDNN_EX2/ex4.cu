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
    size_t imgCounter = 0;
    Mat img = imread("D:\\DataSet\\cifar\\test\\0_cat.png");	// 이미지파일을 읽어 들여 Mat 형식으로 저장시키기
    unsigned char* imgd = img.data;
    size_t imgEleSize = img.elemSize();		// 한픽셀의 실제사이즈, rgb = 3
    size_t imgWidth = img.cols;				// 열 수, 이미지 가로 크기, 입력 데이터의 가로 길이
    size_t imgHeight = img.rows;			// 행의 수, 이미지 세로 크기, 입력 데이터의 세로 길이
    size_t imgChannel = img.channels();		// 깊이, 이미지 채널 수, 입력 데이터의 채널 수
    vector<vector<vector<vector<int>>>> InData_NCHW(10000);

    //입력 데이터 셋팅 N C H W

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
    //  불러온 이미지 데이터 값 확인
    cout << imgCounter << " 번째 이미지 파일 :: ";
	cout << endl;
	for (int c = 0; c < imgChannel; c++)
	{
	for (int y = 0; y < imgHeight; y++)
	{
		for (int x = 0; x < imgWidth; x++)
		{
			cout << setw(3)<< InData_NCHW[imgCounter][c][y][x]<<
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