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
    const int batch_size = 50000; // 이미지 총 갯수
    vector<pair<Mat, string>> ImgBox; // 이미지 데이터, 이미지 이름
    ImgBox = TraverseFilesUsingDFS("D:\\DataSet\\cifar\\train");// 이미지가 저장되어 있는 폴더 경로
    vector<string> LabelBox; // 라벨 정리를 위해 생성
    vector<pair<int, string>> LabelTable; // 라벨링 마다 넘버 부여
    vector<pair<Mat, int>> ImgBox2; // 이미지 데이터, 라벨 넘버

    // 라벨에 번호 부여를 위해 LabelBox 벡터에 값 복사 하고 정렬 및 중복 삭제
    for (int i = 0; i < batch_size; i++)
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
        LabelTable.push_back({ { i }, { LabelBox[i] } });
        //std::cout << "LabelBox :: " << LabelBox[i] << std::endl;// -> 예시 "LabelBox :: truck"
    }

    // LabelTable, 라벨링 확인.
    /*
        for (int i = 0; i < LabelTable.size(); i++)
        {
        std::cout << LabelTable[i].first << " : : " ;
        std::cout << LabelTable[i].second << std::endl; // 9 : : truck
        }
    */

    //ImgBox2 셋팅
    for (int i = 0; i < batch_size; i++)
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

    // 4차 행렬 동적 할당 선언.
    float**** Input = new float** *[batch_size];

    for (int i = 0; i < batch_size; i++)
    {
        Input[i] = new float** [3];

        for (int j = 0; j < 3; j++)
        {
            Input[i][j] = new float*[32];

            for (int k = 0; k < 32; k++)
            {
                Input[i][j][k] = new float[32];
            }
        }
    }

    //입력변수
    const int ImageNum = batch_size;
    const int FeatureNum = 3;
    const int FeatureHeight = 32;
    const int FeatureWidth = 32;

    // mat 형식 - > 4차 행렬
    for (int i = 0; i < ImageNum; i++)
    {
        unsigned char* temp = ImgBox2[i].first.data;

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

    // 이미지 데이터 확인 출력
    std::cout <<
              "*******************" << std::endl <<
              "**Input Data 정보**" << std::endl <<
              "*******************" << std::endl <<
              std::endl;
    std::cout << "Input" << std::endl << std::endl;

    for (int i = 0; i <batch_size; i++)
    {
        std::cout << "라벨 출력 :: " << setw(10) << ImgBox[i].second << "넘버링 출력 :: " << setw(
                      3) << ImgBox2[i].second << "데이터 출력 :: ";

        for (int x = 0; x < 32; x++)
        {
            std::cout << setw(3) << Input[i][0][0][x] ;
        }

        std::cout << std::endl;
    }

    std::cout << std::endl;




























    delete [] Input;
    return 0;
}