#include <mpi.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <dirent.h>
#include <sys/time.h>

using namespace std;
using namespace cv;

#define TAG_INITIAL_REQUEST 1
#define TAG_DONE            2
#define TAG_WORK            3
#define TAG_DIE             4
#define MAX_PATH            1024

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

int main(int argc, char* argv[]) {
    // 【混合平行優化核心】強迫 OpenCV 關閉內部自動多執行緒，將主導權還給 MPI
    cv::setNumThreads(1);

    int rank, size;
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2) {
        if (rank == 0) cout << "Usage: mpirun -np <N> ./parallel_face <folder_path>" << endl;
        MPI_Finalize();
        return 0;
    }

    string target_folder = argv[1];
    string cascade_path = "haarcascade_frontalface_default.xml";

    CascadeClassifier face_cascade;
    if (!face_cascade.load(cascade_path)) {
        cout << "Rank " << rank << " ❌ Error: Cannot load face cascade xml!" << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
        return -1;
    }

    if (rank == 0) {
        vector<string> image_list = get_images_in_folder(target_folder);
        cout << "===== Parallel Processing Start =====" << endl;
        cout << "MPI Size      : " << size << " processes (Pure MPI Controlled Mode)" << endl;
        cout << "Target Folder : " << target_folder << " (" << image_list.size() << " images)" << endl;

        struct timeval start, end;
        gettimeofday(&start, NULL);

        int total_faces = 0;
        int next_image_idx = 0;
        int active_workers = size - 1;

        vector<int> worker_counts(size, 0);

        if (size < 2) {
            cout << "❌ Error: MPI size must be at least 2!" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
            return -1;
        }

        while (active_workers > 0) {
            int local_faces;
            MPI_Status status;
            
            // 專職高效發派，絕不分心算圖
            MPI_Recv(&local_faces, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            int worker_rank = status.MPI_SOURCE;

            if (status.MPI_TAG == TAG_DONE) {
                total_faces += local_faces;
                worker_counts[worker_rank]++;
            }

            if (next_image_idx < image_list.size()) {
                string next_img_path = image_list[next_image_idx++];
                MPI_Send(next_img_path.c_str(), next_img_path.length() + 1, MPI_CHAR, worker_rank, TAG_WORK, MPI_COMM_WORLD);
            } else {
                MPI_Send(NULL, 0, MPI_CHAR, worker_rank, TAG_DIE, MPI_COMM_WORLD);
                active_workers--;
            }
        }

        gettimeofday(&end, NULL);
        double total_time = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000000.0;

        cout << "===== Result =====" << endl;
        cout << "Total Faces Detected : " << total_faces << endl;
        cout << "Total Processing Time: " << total_time << " seconds" << endl;
        cout << "Average FPS          : " << image_list.size() / total_time << endl;

        cout << "\n===== Worker Load Distribution =====" << endl;
        for (int m = 1; m < size; m++) {
            double percentage = (double)worker_counts[m] / image_list.size() * 100.0;
            cout << "Worker " << m << " processed: " << worker_counts[m] << " images (" << percentage << "%)" << endl;
        }
        cout << "====================================" << endl;

    } else {
        int faces_found_last_turn = 0;
        MPI_Send(&faces_found_last_turn, 1, MPI_INT, 0, TAG_INITIAL_REQUEST, MPI_COMM_WORLD);

        while (true) {
            char path_buf[MAX_PATH];
            MPI_Status status;
            MPI_Recv(path_buf, MAX_PATH, MPI_CHAR, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            if (status.MPI_TAG == TAG_DIE) {
                break;
            }

            string img_path(path_buf);
            Mat img = imread(img_path, IMREAD_GRAYSCALE);
            
            if (img.empty()) {
                faces_found_last_turn = 0;
                continue;
            }

            vector<Rect> faces;
            face_cascade.detectMultiScale(img, faces, 1.1, 3, 0, Size(30, 30));
            faces_found_last_turn = faces.size();

            MPI_Send(&faces_found_last_turn, 1, MPI_INT, 0, TAG_DONE, MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();
    return 0;
}
