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
	vector<pair<Mat, string>> ImgBox;					// 가져온 이미지들을 넣을 벡터

	CString ImgFolderPath = ("D:\\DataSet\\cifar\\test");	// 이미지 파일이 들어있는 폴더 경로
	CFileFind finder;									// CFileFind객체 생성
	bool bFind = finder.FindFile(ImgFolderPath + ("\\*.*"));// 파일 경로에 들어 있는 이미지 찾기
	size_t imgCounter = 1;
	while (bFind)
	{


		bFind = finder.FindNextFile();

		if (finder.IsDots())
			continue;

		CString cFileName = finder.GetFileTitle(); // 확장자를 뺀 파일 이름
		CT2CA CstringtoString2(cFileName);		 // Cstring - > string 데이터 형 변환1
		string FileName(CstringtoString2);		 // Cstring - > string 데이터 형 변환2

		CString cstr = finder.GetFileName();	// 파일 이름
		CString ImgPath;
		ImgPath += ImgFolderPath;
		ImgPath += L"\\";
		ImgPath += cstr;


		CT2CA CstringtoString(ImgPath);			// Cstring - > string 데이터 형 변환1
		cv::String sstr(CstringtoString);			// Cstring - > string 데이터 형 변환2
		Mat img = imread(sstr, 1);

		ImgBox.push_back({ { img },{ FileName } });

		//cout << imgCounter << " 번째 이미지 파일 :: ";
		//cout << ImgBox[imgCounter].second << endl; // 파일 이름
		//cout << ImgBox[imgCounter].first << endl; // 픽셀에 저장된 데이터 값
		//imgCounter += 1;

	}


	cout << ImgBox[imgCounter].second << endl; // 파일 이름
	cout << ImgBox[imgCounter].first << endl;  // 픽셀에 저장된 데이터 값

	Mat img = ImgBox[imgCounter].first;

	namedWindow("img", 0); // 0 이면 resizing 가능
	imshow("img", img); //show
	waitKey(0);
	destroyAllWindows();

	return 0;
}