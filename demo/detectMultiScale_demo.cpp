#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>
//#include <opencv2/core/cuda.hpp>
//#include <opencv2/cudaobjdetect.hpp>
//#include <opencv2/cudaimgproc.hpp>
using namespace cv;

int main(int argc, char**argv) {
    std::cout << "OpenCV version=" << std::hex << CV_VERSION << std::dec << std::endl;

    cv::Mat frame;
    cv::UMat uframe, uFrameGray;
    //cv::cuda::GpuMat image_gpu, image_gpu_gray;
    cv::VideoCapture capture;

    bool useOpenCL = (argc >= 2) ? atoi(argv[1]) : false;
    std::cout << "Use OpenCL=" << useOpenCL << std::endl;
    cv::ocl::setUseOpenCL(useOpenCL);

    //bool useCuda = (argc >= 3) ? atoi(argv[2]) : false;
    //std::cout << "Use CUDA=" << useCuda << std::endl;

    cv::Ptr<cv::CascadeClassifier> cascade = cv::makePtr<cv::CascadeClassifier>("/home/ubuntu/WYZ/mess/haarcascade_frontalface_alt.xml");
    //cv::Ptr<cv::cuda::CascadeClassifier> cascade_gpu = cv::cuda::CascadeClassifier::create("data/lbpcascades/lbpcascade_frontalface.xml");

    double time = 0.0;
    int nb = 0;
    capture.open("/home/ubuntu/WYZ/mess/demo/example.mp4");
    namedWindow("Demo", CV_WINDOW_AUTOSIZE);
    if(capture.isOpened()) {
        for(;;) {
            capture >> frame;
	    imshow("Demo", frame);
            if(frame.empty() || nb >= 1000) {
                break;
            }

            std::vector<cv::Rect> faces;
            double t = 0.0;
            //if(!useCuda) {
                t = (double) cv::getTickCount();
                frame.copyTo(uframe);
                cv::cvtColor(uframe, uFrameGray, CV_BGR2GRAY);
                cascade->detectMultiScale(uFrameGray, faces);
                t = ((double) cv::getTickCount() - t) / cv::getTickFrequency();
            // } else {
            //     t = (double) cv::getTickCount();
            //     image_gpu.upload(frame);
            //     cv::cuda::cvtColor(image_gpu, image_gpu_gray, CV_BGR2GRAY);
            //     cv::cuda::GpuMat objbuf;
            //     cascade_gpu->detectMultiScale(image_gpu_gray, objbuf);
            //     cascade_gpu->convert(objbuf, faces);
            //     t = ((double) cv::getTickCount() - t) / cv::getTickFrequency();
            // }

            time += t;
            nb++;

            for(std::vector<cv::Rect>::const_iterator it = faces.begin(); it != faces.end(); ++it) {
                cv::rectangle(frame, *it, cv::Scalar(0,0,255));
            }
            std::stringstream ss;
            ss << "FPS=" << (nb / time);
            cv::putText(frame, ss.str(), cv::Point(30, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0,0,255));

            cv::imshow("Frame", frame);
            char c = cv::waitKey(30);
            if(c == 27) {
                break;
            }
        }
    }
    capture.release();
    waitKey(0);
    std::cout << "Mean time=" << (time / nb) << " s" << " ; Mean FPS=" << (nb / time) << " ; nb=" << nb << std::endl;
    return 0;
}
