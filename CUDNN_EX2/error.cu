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


int main()
{
    const int batch_size = 10;// 이미지 갯수
    const int num_labels = 10; // 라벨링 수

    float OutSoft[batch_size][num_labels]; // 소프트 맥스 결과 값 
    float target[batch_size]; // 정답

    // 임시 값 지정 (soft max 결과 값 )
    for (int j = 0; j < batch_size; j++)
    {
        for (int i = 0; i < num_labels; i++)
        {
            OutSoft[j][i] = 1;
        }
    }
	// one hot 계산을 위해 소프트 맥스 결과에서 출력되는 10개 중 임으로 하나 만 2값이 들어가게 하고 
	// 3개는 정답의 위치와 다르게 나머지 7개는 같게 함 즉 정확도 70% 이 나오게 임의로 값 지정  
    OutSoft[0][1] = 2.0;
    OutSoft[1][2] = 2.0;
    OutSoft[2][3] = 2.0;
    OutSoft[3][3] = 2.0;
    OutSoft[4][4] = 2.0;
    OutSoft[5][5] = 2.0;
    OutSoft[6][6] = 2.0;
    OutSoft[7][7] = 2.0;
    OutSoft[8][8] = 2.0;
    OutSoft[9][9] = 2.0;
	
	// 임시 값 지정 (정답 값)// 0 1,2,3,4,5,6,7,8,9, 0 1,2,3,4,5,6,7,8,9,
    for (int j = 0; j < batch_size; j++)
    {
        target[j] = j % 10;
    }

    //one hot 위치 찾기
    int predicted[batch_size]; // 소프트 맥스 결과 값에서 가장 큰 값이 들어 있는 위치(인덱스)를 저장하는 배열

    for (size_t i = 0; i < batch_size; i++)
    {
        float temp = OutSoft[i][0];// 예측된 첫번째 값을 임시 변수에 저장 
        int indexJ = 0; // 가장 큰값을 찾기 위해 사용 되는 위치 인덱스 

        for (size_t j = 0; j < num_labels-1; j++)
        {
            if (temp > OutSoft[i][j + 1]) // 임시 변수에 넣어준 값과 비교
            {
                OutSoft[i][j + 1] = 0; // 임시 변수에 들어 있는 값보다 작다면 0 입력
            }
            else                      // 임시 변수에 들어 있는 값보다 크다면 
            {
                temp = OutSoft[i][j + 1]; // 임시 변수에 해당 값을 저장
                OutSoft[i][indexJ] = 0; // 임시 변수에 이전에 들어 있던 값의 인덱스를 이용하여 이전 값 위치에 0 입력
                indexJ = j + 1; // 제일 큰 값(정답 예측 값) 위치를 인덱스 변수에 저장
            }
        }

        predicted[i] = indexJ; // 해당 이미지의 소프트 맥스 값중 가장 큰 값을 갖는 위치 인덱스를 배열에 저장
    }

    //accuracy 계산
    int count = 0;
    for (size_t i = 0; i < batch_size; i++)
    {
        if (predicted[i] == target[i])
            count++;
    }
	cout << batch_size << " 개의 이미지들 중 " << count << " 개 맞음"<<endl;
    cout << "정확도 :: " << (count*1.0 / batch_size)*100 << " %"<< endl;
	cout << "에러   :: " << (1-(count*1.0 / batch_size))*100 << " %" << endl;

    return 0;
}