#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "afx.h"
#include <string.h>
#include <iostream>

using namespace std;
using namespace cv;


// img loading  ====>>> vector<pair<Mat, string>> ImgBox;	

int main(int, char)
{
	vector<pair<Mat, string>> ImgBox;					// ������ �̹������� ���� ����

	CString ImgFolderPath = ("D:\\DataSet\\cifar\\test");	// �̹��� ������ ����ִ� ���� ���
	CFileFind finder;									// CFileFind��ü ����
	bool bFind = finder.FindFile(ImgFolderPath + ("\\*.*"));// ���� ��ο� ��� �ִ� �̹��� ã��
	size_t imgCounter = 1;
	while (bFind)
	{


		bFind = finder.FindNextFile();

		if (finder.IsDots())
			continue;

		CString cFileName = finder.GetFileTitle(); // Ȯ���ڸ� �� ���� �̸�
		CT2CA CstringtoString2(cFileName);		 // Cstring - > string ������ �� ��ȯ1
		string FileName(CstringtoString2);		 // Cstring - > string ������ �� ��ȯ2

		CString cstr = finder.GetFileName();	// ���� �̸�
		CString ImgPath;
		ImgPath += ImgFolderPath;
		ImgPath += L"\\";
		ImgPath += cstr;


		CT2CA CstringtoString(ImgPath);			// Cstring - > string ������ �� ��ȯ1
		cv::String sstr(CstringtoString);			// Cstring - > string ������ �� ��ȯ2
		Mat img = imread(sstr, 1);

		ImgBox.push_back({ { img },{ FileName } });

		//cout << imgCounter << " ��° �̹��� ���� :: ";
		//cout << ImgBox[imgCounter].second << endl; // ���� �̸�
		//cout << ImgBox[imgCounter].first << endl; // �ȼ��� ����� ������ ��
		//imgCounter += 1;

	}


	cout << ImgBox[imgCounter].second << endl; // ���� �̸�
	cout << ImgBox[imgCounter].first << endl;  // �ȼ��� ����� ������ ��

	Mat img = ImgBox[imgCounter].first;

	namedWindow("img", 0); // 0 �̸� resizing ����
	imshow("img", img); //show
	waitKey(0);
	destroyAllWindows();

	return 0;
}