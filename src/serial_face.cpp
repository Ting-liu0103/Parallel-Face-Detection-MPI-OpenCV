#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <dirent.h>
#include <sys/time.h>

using namespace cv;
using namespace std;

// 只讀取資料夾中的 .jpg / .png 圖片
vector<string> get_images_in_folder(const string& folder) {
    vector<string> images;
    DIR* dir = opendir(folder.c_str());
    if (!dir) return images;

    struct dirent* entry;
    while ((entry = readdir(dir)) != NULL) {
        string filename = entry->d_name;
        if (filename.find(".jpg") != string::npos || filename.find(".png") != string::npos) {
            images.push_back(folder + "/" + filename);
        }
    }

    closedir(dir);
    return images;
}

int main(int argc, char** argv) {
    // 關閉 OpenCV 內部自動多執行緒，確保 baseline 絕對公平
    cv::setNumThreads(1);

    string test_folder = "dataset/high_load";
    if (argc > 1) {
        test_folder = argv[1];
    }

    string cascade_path = "haarcascade_frontalface_default.xml";

    CascadeClassifier face_cascade;
    if (!face_cascade.load(cascade_path)) {
        cout << "Error: Cannot load face cascade xml!" << endl;
        return -1;
    }

    vector<string> image_paths = get_images_in_folder(test_folder);
    if (image_paths.empty()) {
        cout << "No images found in " << test_folder << endl;
        return 0;
    }

    cout << "===== Serial Processing Start =====" << endl;
    cout << "Target Folder : " << test_folder << " (" << image_paths.size() << " images)" << endl;

    int total_faces = 0;

    struct timeval start, end;
    gettimeofday(&start, NULL);

    for (size_t i = 0; i < image_paths.size(); i++) {
        // 直接以灰階載入，與平行版完全對齊解碼精度！
        Mat img = imread(image_paths[i], IMREAD_GRAYSCALE);

        if (img.empty()) {
            continue;
        }

        vector<Rect> faces;
        face_cascade.detectMultiScale(img, faces, 1.1, 3, 0, Size(30, 30));

        total_faces += faces.size();
    }

    gettimeofday(&end, NULL);
    double total_time = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000000.0;

    cout << "===== Result =====" << endl;
    cout << "Total Faces Detected : " << total_faces << endl;
    cout << "Total Processing Time: " << total_time << " seconds" << endl;
    cout << "Average FPS          : " << image_paths.size() / total_time << endl;

    return 0;
}
