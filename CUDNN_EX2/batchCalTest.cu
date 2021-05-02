#include<io.h>
#include<iostream>
#include<string>
#include<vector>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

//이미지 데이터 및 이미지 이름 가져오기
vector<pair<Mat, string>> TraverseFilesUsingDFS(const string& folder_path)
{
	_finddata_t file_info;
	string any_file_pattern = folder_path + "\\*";
	intptr_t handle = _findfirst(any_file_pattern.c_str(), &file_info);
	vector<pair<Mat, string>> ImgBox;

	//If folder_path exsist, using any_file_pattern will find at least two files "." and "..",
	//of which "." means current dir and ".." means parent dir
	if (handle == -1)
	{
		cerr << "folder path not exist: " << folder_path << endl;
		exit(-1);
	}

	//iteratively check each file or sub_directory in current folder
	do
	{
		string file_name = file_info.name; //from char array to string

										   //check whtether it is a sub direcotry or a file
		if (file_info.attrib & _A_SUBDIR)
		{
			if (file_name != "." && file_name != "..")
			{
				string sub_folder_path = folder_path + "\\" + file_name;
				TraverseFilesUsingDFS(sub_folder_path);
				cout << "a sub_folder path: " << sub_folder_path << endl;
			}
		}
		else  //cout << "file name: " << file_name << endl;
		{
			size_t npo1 = file_name.find('_') + 1;
			size_t npo2 = file_name.find('.');
			size_t npo3 = npo2 - npo1;
			string newname = file_name.substr(npo1, npo3);
			string sub_folder_path2 = folder_path + "\\" + file_name;
			Mat img = imread(sub_folder_path2);
			ImgBox.push_back({ { img },{ newname } });
		}
	} while (_findnext(handle, &file_info) == 0);

	//
	_findclose(handle);
	return ImgBox;
}

int main()
{
	const int ImageNum = 10; // 이미지 총 갯수
	vector<pair<Mat, string>> ImgBox; // 이미지 데이터, 이미지 이름
	ImgBox = TraverseFilesUsingDFS("D:\\DataSet\\cifar\\train");// 이미지가 저장되어 있는 폴더 경로
	vector<string> LabelBox; // 라벨 정리를 위해 생성
	vector<pair<int, string>> LabelTable; // 라벨링 마다 넘버 부여
	float* target_train = new float[ImageNum]; // target 값 , 라벨에 따른 지정된 넘버 값이 담긴 배열 


	// 라벨에 번호 부여를 위해 LabelBox 벡터에 값 복사 하고 정렬 및 중복 삭제
	for (int i = 0; i < ImageNum; i++)
	{
		//std::cout<< "라벨 출력 :: " << ImgBox[i].second << std::endl; // 입력받은순서대로 라벨 출력 -> 예시 "라벨 출력 :: automobile"
		LabelBox.push_back(ImgBox[i].second);
	}

	sort(LabelBox.begin(), LabelBox.end());
	LabelBox.erase(unique(LabelBox.begin(), LabelBox.end()), LabelBox.end());
	int nLabelBoxSize = LabelBox.size();

	// 라벨 번호 부여
	for (int i = 0; i < nLabelBoxSize; i++)
	{
		LabelTable.push_back({ { i },{ LabelBox[i] } });
		//std::cout << "LabelBox :: " << LabelBox[i] << std::endl;// -> 예시 "LabelBox :: truck"
	}

	//target 셋팅
	for (int i = 0; i < ImageNum; i++)
	{
		for (int j = 0; j < LabelTable.size(); j++)
		{
			if (ImgBox[i].second == LabelTable[j].second)
			{
				target_train[i] = LabelTable[j].first;
			}
		}
	}


	//입력변수
	const int batch_size = 2;
	const int FeatureNum = 3;
	const int FeatureHeight = 32;
	const int FeatureWidth = 32;


	// 4차 행렬 동적 할당 선언.
	float**** Input_train = new float** *[ImageNum];

	for (int i = 0; i < ImageNum; i++)
	{
		Input_train[i] = new float**[FeatureNum];

		for (int j = 0; j < FeatureNum; j++)
		{
			Input_train[i][j] = new float*[FeatureHeight];

			for (int k = 0; k < FeatureHeight; k++)
			{
				Input_train[i][j][k] = new float[FeatureWidth];
			}
		}
	}




	for (int a = 0; a < ImageNum / batch_size; a++) { // 배치 계산 루프 


		for (int i = batch_size*a; i < batch_size*(a + 1); i++)
		{
			unsigned char* temp = ImgBox[i].first.data;

			for (int c = 0; c < FeatureNum; c++)
			{
				for (int y = 0; y < FeatureHeight; y++)
				{
					for (int x = 0; x < FeatureWidth; x++)
					{
						Input_train[i][c][y][x] = temp[3 * 32 * y + 3 * x + c];
						
					}
				}
			}

		}

		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		//GPU train 계산 들어갈 자리 


		// 배치 마다 계산 이미지 데이터 확인 출력 (계산 확인을 위해 작성한 부분)
		std::cout <<
			"**************************" << std::endl <<
			"**" << a << "배치의 train set Data 정보**" << std::endl <<
			"**************************" << std::endl <<
			std::endl;
		std::cout << "Input" << std::endl << std::endl;

		for (int i = batch_size*a; i < batch_size*(a + 1); i++)
		{
			std::cout << "Batch 번호:: " << a << " 라벨 출력:: " << setw(9) << ImgBox[i].second << "넘버링 출력 :: " << setw(3) << target_train[i] << "데이터 출력 :: ";

			for (int x = 0; x < 32; x++)
			{
				std::cout << setw(3) << Input_train[i][0][0][x];
			}

			std::cout << std::endl;
		}
		std::cout << std::endl;

		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


	}

	// train 데이터용 데이터 셋에 할당된 메모리 해제
	delete[] target_train;
	delete[] Input_train;
	///////////////////////////////////////////
	//test 계산용 데이터 셋 준비 

	const int ImageNum_test = 10; // 이미지 총 갯수
	vector<pair<Mat, string>> ImgBox_test; // 이미지 데이터, 이미지 이름
	ImgBox_test = TraverseFilesUsingDFS("D:\\DataSet\\cifar\\test");// 이미지가 저장되어 있는 폴더 경로
	vector<string> LabelBox_test; // 라벨 정리를 위해 생성
	vector<pair<int, string>> LabelTable_test; // 라벨링 마다 넘버 부여
	float* target_test = new float[ImageNum_test]; // target 값 , 라벨에 따른 지정된 넘버 값이 담긴 배열 


												   // 라벨에 번호 부여를 위해 LabelBox 벡터에 값 복사 하고 정렬 및 중복 삭제
	for (int i = 0; i < ImageNum_test; i++)
	{
		//std::cout<< "라벨 출력 :: " << ImgBox[i].second << std::endl; // 입력받은순서대로 라벨 출력 -> 예시 "라벨 출력 :: automobile"
		LabelBox_test.push_back(ImgBox_test[i].second);
	}

	sort(LabelBox_test.begin(), LabelBox_test.end());
	LabelBox_test.erase(unique(LabelBox_test.begin(), LabelBox_test.end()), LabelBox_test.end());
	int nLabelBoxSize_test = LabelBox_test.size();

	// 라벨 번호 부여
	for (int i = 0; i < nLabelBoxSize_test; i++)
	{
		LabelTable_test.push_back({ { i },{ LabelBox_test[i] } });
		//std::cout << "LabelBox :: " << LabelBox[i] << std::endl;// -> 예시 "LabelBox :: truck"
	}

	//target 셋팅
	for (int i = 0; i < ImageNum_test; i++)
	{
		for (int j = 0; j < LabelTable_test.size(); j++)
		{
			if (ImgBox_test[i].second == LabelTable_test[j].second)
			{
				target_test[i] = LabelTable_test[j].first;
			}
		}
	}


	//입력변수
	const int batch_size_test = 2;
	const int FeatureNum_test = 3;
	const int FeatureHeight_test = 32;
	const int FeatureWidth_test = 32;


	// 4차 행렬 동적 할당 선언.
	float**** Input_test = new float** *[ImageNum_test];

	for (int i = 0; i < ImageNum_test; i++)
	{
		Input_test[i] = new float**[FeatureNum_test];

		for (int j = 0; j < FeatureNum_test; j++)
		{
			Input_test[i][j] = new float*[FeatureHeight_test];

			for (int k = 0; k < FeatureHeight_test; k++)
			{
				Input_test[i][j][k] = new float[FeatureWidth_test];
			}
		}
	}



	string str = "a";
	for (int a = 0; a < ImageNum_test / batch_size_test; a++) { // 배치 계산 루프 


		for (int i = batch_size_test*a; i < batch_size_test*(a + 1); i++)
		{
			unsigned char* temp = ImgBox_test[i].first.data;

			for (int c = 0; c < FeatureNum_test; c++)
			{
				for (int y = 0; y < FeatureHeight_test; y++)
				{
					for (int x = 0; x < FeatureWidth_test; x++)
					{
						Input_test[i][c][y][x] = temp[3 * 32 * y + 3 * x + c];
						
					}
				}
			}

		}

		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		//GPU test 계산 들어갈 자리 


		// 배치 마다 계산 이미지 데이터 확인 출력 (계산 확인을 위해 작성한 부분)
		std::cout <<
			"**************************" << std::endl <<
			"**" << a << "배치의 test set Data 정보**" << std::endl <<
			"**************************" << std::endl <<
			std::endl;
		std::cout << "Input" << std::endl << std::endl;

		for (int i = batch_size_test*a; i < batch_size_test*(a + 1); i++)
		{
			std::cout << "Batch 번호:: " << a << " 라벨 출력:: " << setw(9) << ImgBox_test[i].second << "넘버링 출력 :: " << setw(3) << target_test[i] << "데이터 출력 :: ";

			for (int x = 0; x < 32; x++)
			{
				std::cout << setw(3) << Input_test[i][0][0][x];
			}

			std::cout << std::endl;
		}
		std::cout << std::endl;

		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


	}




	///////////////////////////////////////////
	//



	






	// test 데이터용 데이터 셋에 할당된 메모리 해제
	delete[] target_test;
	delete[] Input_test;

	return 0;
}