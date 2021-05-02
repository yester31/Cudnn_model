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
    const int batch_size = 10;// �̹��� ����
    const int num_labels = 10; // �󺧸� ��

    float OutSoft[batch_size][num_labels]; // ����Ʈ �ƽ� ��� �� 
    float target[batch_size]; // ����

    // �ӽ� �� ���� (soft max ��� �� )
    for (int j = 0; j < batch_size; j++)
    {
        for (int i = 0; i < num_labels; i++)
        {
            OutSoft[j][i] = 1;
        }
    }
	// one hot ����� ���� ����Ʈ �ƽ� ������� ��µǴ� 10�� �� ������ �ϳ� �� 2���� ���� �ϰ� 
	// 3���� ������ ��ġ�� �ٸ��� ������ 7���� ���� �� �� ��Ȯ�� 70% �� ������ ���Ƿ� �� ����  
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
	
	// �ӽ� �� ���� (���� ��)// 0 1,2,3,4,5,6,7,8,9, 0 1,2,3,4,5,6,7,8,9,
    for (int j = 0; j < batch_size; j++)
    {
        target[j] = j % 10;
    }

    //one hot ��ġ ã��
    int predicted[batch_size]; // ����Ʈ �ƽ� ��� ������ ���� ū ���� ��� �ִ� ��ġ(�ε���)�� �����ϴ� �迭

    for (size_t i = 0; i < batch_size; i++)
    {
        float temp = OutSoft[i][0];// ������ ù��° ���� �ӽ� ������ ���� 
        int indexJ = 0; // ���� ū���� ã�� ���� ��� �Ǵ� ��ġ �ε��� 

        for (size_t j = 0; j < num_labels-1; j++)
        {
            if (temp > OutSoft[i][j + 1]) // �ӽ� ������ �־��� ���� ��
            {
                OutSoft[i][j + 1] = 0; // �ӽ� ������ ��� �ִ� ������ �۴ٸ� 0 �Է�
            }
            else                      // �ӽ� ������ ��� �ִ� ������ ũ�ٸ� 
            {
                temp = OutSoft[i][j + 1]; // �ӽ� ������ �ش� ���� ����
                OutSoft[i][indexJ] = 0; // �ӽ� ������ ������ ��� �ִ� ���� �ε����� �̿��Ͽ� ���� �� ��ġ�� 0 �Է�
                indexJ = j + 1; // ���� ū ��(���� ���� ��) ��ġ�� �ε��� ������ ����
            }
        }

        predicted[i] = indexJ; // �ش� �̹����� ����Ʈ �ƽ� ���� ���� ū ���� ���� ��ġ �ε����� �迭�� ����
    }

    //accuracy ���
    int count = 0;
    for (size_t i = 0; i < batch_size; i++)
    {
        if (predicted[i] == target[i])
            count++;
    }
	cout << batch_size << " ���� �̹����� �� " << count << " �� ����"<<endl;
    cout << "��Ȯ�� :: " << (count*1.0 / batch_size)*100 << " %"<< endl;
	cout << "����   :: " << (1-(count*1.0 / batch_size))*100 << " %" << endl;

    return 0;
}