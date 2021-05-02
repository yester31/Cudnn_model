#include<io.h>
#include<iostream>
#include<string>
#include<vector>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

//�̹��� ������ �� �̹��� �̸� ��������
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
	const int ImageNum = 1000; // �̹��� �� ����
	vector<pair<Mat, string>> ImgBox; // �̹��� ������, �̹��� �̸�
	ImgBox = TraverseFilesUsingDFS("D:\\DataSet\\cifar\\train");// �̹����� ����Ǿ� �ִ� ���� ���
	vector<string> LabelBox; // �� ������ ���� ����
	vector<pair<int, string>> LabelTable; // �󺧸� ���� �ѹ� �ο�
	float* target = new float[ImageNum]; // target �� , �󺧿� ���� ������ �ѹ� ���� ��� �迭 
	

	// �󺧿� ��ȣ �ο��� ���� LabelBox ���Ϳ� �� ���� �ϰ� ���� �� �ߺ� ����
	for (int i = 0; i < ImageNum; i++)
	{
		//std::cout<< "�� ��� :: " << ImgBox[i].second << std::endl; // �Է¹���������� �� ��� -> ���� "�� ��� :: automobile"
		LabelBox.push_back(ImgBox[i].second);
	}

	sort(LabelBox.begin(), LabelBox.end());
	LabelBox.erase(unique(LabelBox.begin(), LabelBox.end()), LabelBox.end());
	int nLabelBoxSize = LabelBox.size();

	// �� ��ȣ �ο�
	for (int i = 0; i < nLabelBoxSize; i++)
	{
		LabelTable.push_back({ { i },{ LabelBox[i] } });
		//std::cout << "LabelBox :: " << LabelBox[i] << std::endl;// -> ���� "LabelBox :: truck"
	}

	//target ����
	for (int i = 0; i < ImageNum; i++)
	{
		for (int j = 0; j < LabelTable.size(); j++)
		{
			if (ImgBox[i].second == LabelTable[j].second)
			{
				target[i] = LabelTable[j].first;
			}
		}
	}


	//�Էº���
	const int batch_size = 100;
	const int FeatureNum = 3;
	const int FeatureHeight = 32;
	const int FeatureWidth = 32;


	// 4�� ��� ���� �Ҵ� ����.
	float**** Input = new float** *[ImageNum];

	for (int i = 0; i < ImageNum; i++)
	{
		Input[i] = new float**[FeatureNum];

		for (int j = 0; j < FeatureNum; j++)
		{
			Input[i][j] = new float*[FeatureHeight];

			for (int k = 0; k < FeatureHeight; k++)
			{
				Input[i][j][k] = new float[FeatureWidth];
			}
		}
	}




	for (int a = 0; a < ImageNum/batch_size; a++) { // 10�� ���ư��� 
	
		
		for (int i = batch_size*a; i < batch_size*(a+1) ; i++)
		{
			unsigned char* temp = ImgBox[i].first.data;

			for (int c = 0; c < FeatureNum; c++)
			{
				for (int y = 0; y < FeatureHeight; y++)
				{
					for (int x = 0; x < FeatureWidth; x++)
					{
						Input[i][c][y][x] = temp[3 * 32 * y + 3 * x + c];
					}
				}
			}

		}

		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		//GPU ��� �� �ڸ� 


		// ��ġ ���� ��� �̹��� ������ Ȯ�� ��� (��� Ȯ���� ���� �ۼ��� �κ�)
		std::cout <<
			"**************************" << std::endl <<
			"**"<< a << "��ġ�� image Data ����**" << std::endl <<
			"**************************" << std::endl <<
			std::endl;
		std::cout << "Input" << std::endl << std::endl;

		for (int i = batch_size*a; i < batch_size*(a + 1); i++)
		{
			std::cout << "Batch ��ȣ:: " << a << " �� ���:: " << setw(9) << ImgBox[i].second << "�ѹ��� ��� :: " << setw(3) << target[i] << "������ ��� :: ";

			for (int x = 0; x < 32; x++)
			{
				std::cout << setw(3) << Input[i][0][0][x];
			}

			std::cout << std::endl;
		}
		std::cout << std::endl;

		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	
	}


	




	delete[] Input;
	return 0;
}