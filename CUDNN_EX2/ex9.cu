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

            ImgBox.push_back({ { img }, { newname } });
        }
    }
    while (_findnext(handle, &file_info) == 0);

    //
    _findclose(handle);
    return ImgBox;
}

int main()
{
    const int numImgs = 10000; // �̹��� �� ����
    string folder_path = "D:\\DataSet\\cifar\\test"; // �̹����� ����Ǿ� �ִ� ���� ���
    vector<pair<Mat, string>> ImgBox; // �̹��� ������, �̹��� �̸� 
	ImgBox = TraverseFilesUsingDFS(folder_path);
    vector<string> LabelBox; // �� ������ ���� ����
    vector<pair<int, string>> LabelTable; // �󺧸� ���� �ѹ� �ο�
    vector<pair<Mat, int>> ImgBox2; // �̹��� ������, �� �ѹ�
    vector<vector<int>> TargetY; // �� �ѹ� -> ������ ���� �����ͷ� ����
   
	// �󺧿� ��ȣ �ο��� ���� LabelBox ���Ϳ� �� ���� �ϰ� ���� �� �ߺ� ����
    for (int i = 0; i < 10000; i++)
    {
        //std::cout << ImgBox[i].second << std::endl;
        LabelBox.push_back(ImgBox[i].second);
    }
    sort(LabelBox.begin(), LabelBox.end());
    LabelBox.erase(unique(LabelBox.begin(), LabelBox.end()), LabelBox.end());
    int nLabelBoxSize =  LabelBox.size();

	// �� ��ȣ �ο�
    for (int i = 0; i < nLabelBoxSize; i++) 
    {
        LabelTable.push_back({ { i }, { LabelBox[i] } });
        //std::cout << LabelBox[i] << std::endl;
    }

	// LabelTable, �󺧸� Ȯ��.
    /*
        for (int i = 0; i < LabelTable.size(); i++)
        {
        std::cout << LabelTable[i].first << " : : " ;
        std::cout << LabelTable[i].second << std::endl;
        } 
    */

	//ImgBox2 ����
    for (int i = 0; i < numImgs; i++)
    {
        ImgBox2.push_back({ ImgBox[i].first, 0 });

        for (int j = 0; j < LabelTable.size(); j++)
        {
            if (ImgBox[i].second == LabelTable[j].second)
            {
                ImgBox2[i].second = LabelTable[j].first;
            }
        }
    }								

	// TargetY ����, ���� ������ ���·� ǥ��
    TargetY.resize(numImgs);
    for (int i = 0; i < numImgs; i++)
    {
        TargetY[i].resize(nLabelBoxSize, 0);
    }
    for (int i = 0; i < numImgs; i++)
    {
        int idx = ImgBox2[i].second;
        TargetY[i][idx] = 1;
    }

	// ����� ����ȭ ���·� ǥ���� ���� Ȯ�� ���
	
	for (int i = 0; i < numImgs; i++)
	{
		std::cout << ImgBox2[i].second << " :: ";

		for (int j = 0; j < nLabelBoxSize; j++)
		{
			cout << TargetY[i][j] << ", ";
		}

		cout << endl;
	}
	 

	// 4�� ��� ���� �Ҵ� ����.
	int **** Input = new int***[10000];
	for (int i = 0; i < 10000; i++)
	{
		Input[i] = new int**[3];
		for (int j = 0; j < 3; j++)
		{
			Input[i][j] = new int*[32];
			for (int k = 0; k < 32; k++)
			{
				Input[i][j][k] = new int[32];
			}
		}
	}

	// mat ���� - > 4�� ��� 
    for (int i = 0; i < 10000; i++)
    {
		unsigned char* temp = ImgBox2[i].first.data;
        for (int c = 0; c < 3; c++)
        {
            for (int y = 0; y < 32; y++)
            {
                for (int x = 0; x < 32; x++)
                {
					Input[i][c][y][x] = temp[3 * 32 * y + 3 * x + c];
                }
            }
        }
    }

	// �̹��� ������ Ȯ�� ���
	/*
    std::cout <<
              "*******************" << std::endl <<
              "**Input Data ����**" << std::endl <<
              "*******************" << std::endl <<
    std::endl;
    std::cout << "Input" << std::endl << std::endl;
	
    for (int i = 0; i < 10000; i++)
    {
		for (int x = 0; x < 32; x++)
		{
			std::cout << setw(3) << Input[i][0][0][x] ;
		}
		std::cout << std::endl;
    }
    std::cout << std::endl;
	*/
	
    return 0;
}