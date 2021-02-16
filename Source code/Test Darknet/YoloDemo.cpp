#include <iostream>
#include <queue>
#include <iterator>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include "INIReader.h"

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/all_layers.hpp>
#include <opencv2/core/utils/filesystem.hpp>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

//constexpr float CONFIDENCE_THRESHOLD = 0;
//constexpr float NMS_THRESHOLD = 0.4;
//constexpr int NUM_CLASSES = 80;

// colors for bounding boxes
const cv::Scalar colors[] = {
    {0, 255, 255},
    {255, 255, 0},
    {0, 255, 0},
    {255, 0, 0}
};
const auto NUM_COLORS = sizeof(colors) / sizeof(colors[0]);

void Yolo(cv::Mat frame, int NUM_CLASSES, int CONFIDENCE_THRESHOLD, int NMS_THRESHOLD, std::vector<cv::Mat> detections, std::vector<std::vector<int>>& indices, std::vector<std::vector<cv::Rect>>& boxes, std::vector<std::vector<float>>& scores)
{
    //detect
    for (auto& output : detections)
    {
        const auto num_boxes = output.rows;
        //std::cout << num_boxes << std::endl;
        for (int i = 0; i < num_boxes; ++i)
        {
            auto x = output.at<float>(i, 0) * frame.cols;
            auto y = output.at<float>(i, 1) * frame.rows;
            auto width = output.at<float>(i, 2) * frame.cols;
            auto height = output.at<float>(i, 3) * frame.rows;
            cv::Rect rect(x - width / 2, y - height / 2, width, height);

            for (int c = 0; c < NUM_CLASSES; ++c)
            {
                auto confidence = *output.ptr<float>(i, 5 + c);
                if (confidence >= CONFIDENCE_THRESHOLD)
                {
                    boxes[c].push_back(rect);
                    scores[c].push_back(confidence);
                }
            }
        }
    }

    //non-maximum suppress
    for (int c = 0; c < NUM_CLASSES; ++c)
    {
        //std::cout << "Size before NMS: " << boxes[c].size() << std::endl;
        cv::dnn::NMSBoxes(boxes[c], scores[c], 0.0, NMS_THRESHOLD, indices[c]);
        //std::cout << "Size after NMS: " << boxes[c].size() << std::endl;
    }
}

cv::Rect Render(cv::Mat frame, int c, int i, std::vector<std::string> class_names,std::vector<std::vector<int>> indices, std::vector<std::vector<cv::Rect>> boxes, std::vector<std::vector<float>> scores, std::ofstream& log_file, int rWidth = 0, int rHeight = 0)
{
    const auto color = colors[c % NUM_COLORS];

    auto idx = indices[c][i];
    auto& rect = boxes[c][idx];
    rect.x += rWidth;
    rect.y += rHeight;
    if (rWidth != 0 && rHeight != 0)
        log_file << "Ping: " << rect.x << " " << rect.y << std::endl;
    std::ostringstream label_ss;
    label_ss << class_names[c] << ": " << std::fixed << std::setprecision(2) << scores[c][idx];

    auto label = label_ss.str();

    int baseline;
    auto label_bg_sz = cv::getTextSize(label.c_str(), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, 1, &baseline);

    cv::rectangle(frame, cv::Point(rect.x, rect.y), cv::Point(rect.x + rect.width, rect.y + rect.height), color, 3);

    if (rect.y - label_bg_sz.height - baseline < 10)
    {
        //label.c_str()
        cv::rectangle(frame, cv::Point(rect.x + rect.width - label_bg_sz.width, rect.y + rect.height - label_bg_sz.height - baseline - 10), cv::Point(rect.x + rect.width, rect.y + rect.height), color, cv::FILLED);
        cv::putText(frame, label.c_str(), cv::Point(rect.x + rect.width - label_bg_sz.width, rect.y + rect.height - baseline - 5), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 0, 0));
    }
    else
    {
        cv::rectangle(frame, cv::Point(rect.x, rect.y - label_bg_sz.height - baseline - 10), cv::Point(rect.x + label_bg_sz.width, rect.y), color, cv::FILLED);
        cv::putText(frame, label.c_str(), cv::Point(rect.x, rect.y - baseline - 5), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 0, 0));
    }
    return boxes[c][idx];
}

int main()
{
    INIReader ini("config.ini");
    int NUM_CLASSES = 0;
    std::vector<std::string> class_names, class_names_LP;
    float CONFIDENCE_THRESHOLD = ini.GetReal("PARAMS", "confidence_threshold", 0);
    float NMS_THRESHOLD = ini.GetReal("PARAMS", "non_maximum_suppresion_threshold", 0.4);
    //{ 
    cv::String classPath = ini.GetString("PRIMARY", "class", "classes.txt"); //"model.names";
    std::ifstream class_file(classPath);
    if (!class_file)
    {
        std::cerr << "failed to open classes.txt\n";
        return 0;    
    }

    std::string line;
    while (std::getline(class_file, line))
    {
        class_names.push_back(line);
        ++NUM_CLASSES;
    }

    int NUM_CLASSES_LP = 0;

    cv::String classPath_LP = ini.GetString("SECONDARY", "class", "classes.txt"); //"model.names";
    std::ifstream class_file_LP(classPath_LP);
    if (!class_file_LP)
    {
        std::cerr << "failed to open classes.txt\n";
        return 0;
    }

    //std::string line;
    while (std::getline(class_file_LP, line))
    {
        class_names_LP.push_back(line);
        ++NUM_CLASSES_LP;
    }
    //}
    cv::String logPath = ini.GetString("FILE", "log", "logs.txt");
    std::ofstream log_file(logPath);
    //cv::String stream = "rtsp://admin:!iotcamera3@192.168.50.24/trackID=1";
    //cv::String offline = "truck_3.mp4";
    //cv::String videoPath = stream;

    
    //if (source.isOpened() == false) {
    //    std::cerr << "Open Error" << std::endl;
    //    return -1;
    //}
    //cv::VideoCapture source("rtsp://admin:!iotcamera3@192.168.50.24/trackID=1");
    //double dWidth = source.get(cv::CAP_PROP_FRAME_WIDTH); //get the width of frames of the source video
    //double dHeight = source.get(cv::CAP_PROP_FRAME_HEIGHT); //get the height of frames of the source video
    //double dFPS = source.get(cv::CAP_PROP_FPS); //get the frame rate of the source video
    //cv::VideoWriter out("out.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), dFPS, cv::Size(dWidth, dHeight));// "output.avi", cv::CAP_PROP_FOURCC(), )
    cv::String cfgPath = ini.GetString("PRIMARY", "config", ""); //"model.cfg";
    cv::String weightPath = ini.GetString("PRIMARY", "weight", ""); //"model.weights";
    auto net = cv::dnn::readNetFromDarknet(cfgPath, weightPath);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
    //net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    //net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    auto output_names = net.getUnconnectedOutLayersNames();

    cv::String cfgPath_LP = ini.GetString("SECONDARY", "config", ""); //"model.cfg";
    cv::String weightPath_LP = ini.GetString("SECONDARY", "weight", ""); //"model.weights";
    auto net_LP = cv::dnn::readNetFromDarknet(cfgPath, weightPath);
    net_LP.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net_LP.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
    auto output_names_LP = net_LP.getUnconnectedOutLayersNames();

    cv::Mat frame, blob, blob_LP;
    std::vector<cv::Mat> detections, detections_LP;

    // Get all image in the folder
    std::vector<cv::String> filenames;
    cv::String imgPath = ini.GetString("FILE", "input", ""); //"//192.168.0.199/ubay_share/Container data/Truck_2/*.png
    cv::String outPath = ini.GetString("FILE", "output", "");
    cv::String groundtruthPath = outPath + "/ground_truth/";
    cv::glob(imgPath, filenames);
    cv::utils::fs::createDirectory(outPath);
    cv::utils::fs::createDirectory(groundtruthPath);
    //cv::VideoCapture source(imgPath);

    size_t nFiles = filenames.size();
    log_file << "The number of images is " << nFiles << std::endl;
    //while (cv::waitKey(1) < 1)
    //for(int i = 0; i < 6000; ++i)
    for(size_t i = 0; i < nFiles && cv::waitKey(1) < 1; ++i)
    {
        log_file << filenames[i] << std::endl;
        frame = cv::imread(filenames[i]);
        //source >> frame;
        if (frame.empty())
        {
            cv::waitKey();
            break;
        }

        auto total_start = std::chrono::steady_clock::now();
        cv::dnn::blobFromImage(frame, blob, 0.00392, cv::Size(608, 608), cv::Scalar(), true, false, CV_32F);
        net.setInput(blob);

        auto dnn_start = std::chrono::steady_clock::now();
        try
        {
            net.forward(detections, output_names);
        }
        catch (std::exception& ex)
        {
            log_file << "We got a problem!: " << ex.what();
            break;
        }
        auto dnn_end = std::chrono::steady_clock::now();

        std::vector<std::vector<int>> indices(NUM_CLASSES);
        std::vector<std::vector<cv::Rect>> boxes(NUM_CLASSES); //bounding boxes
        std::vector<std::vector<float>> scores(NUM_CLASSES); //confidence scores

        std::vector<std::vector<int>> indices_LP(NUM_CLASSES_LP);
        std::vector<std::vector<cv::Rect>> boxes_LP(NUM_CLASSES_LP); //bounding boxes
        std::vector<std::vector<float>> scores_LP(NUM_CLASSES_LP); //confidence scores
        //std::vector<int> indices[NUM_CLASSES];
        //std::vector<cv::Rect> boxes[NUM_CLASSES];
        //std::vector<float> scores[NUM_CLASSES];

        //if (detections.size() != 0)
        //std::cout << filenames[i] << std::endl;

        Yolo(frame, NUM_CLASSES, CONFIDENCE_THRESHOLD, NMS_THRESHOLD, detections, indices, boxes, scores);
        //detect
        //for (auto& output : detections)
        //{
        //    const auto num_boxes = output.rows; 
        //    //std::cout << num_boxes << std::endl;
        //    for (int i = 0; i < num_boxes; ++i)
        //    {
        //        auto x = output.at<float>(i, 0) * frame.cols;
        //        auto y = output.at<float>(i, 1) * frame.rows;
        //        auto width = output.at<float>(i, 2) * frame.cols;
        //        auto height = output.at<float>(i, 3) * frame.rows;
        //        cv::Rect rect(x - width / 2, y - height / 2, width, height);

        //        for (int c = 0; c < NUM_CLASSES; ++c)
        //        {
        //            auto confidence = *output.ptr<float>(i, 5 + c);
        //            if (confidence >= CONFIDENCE_THRESHOLD)
        //            {
        //                boxes[c].push_back(rect);
        //                scores[c].push_back(confidence);
        //            }
        //        }
        //    }
        //}

        ////non-maximum suppress
        //for (int c = 0; c < NUM_CLASSES; ++c)
        //{
        //    //std::cout << "Size before NMS: " << boxes[c].size() << std::endl;
        //    cv::dnn::NMSBoxes(boxes[c], scores[c], 0.0, NMS_THRESHOLD, indices[c]);
        //    //std::cout << "Size after NMS: " << boxes[c].size() << std::endl;
        //}

        //label classes
        for (int c = 0; c < NUM_CLASSES; ++c)
        {
            //if (c != 2 && c != 5 && c != 7)
            //    continue;
            //if (c != 7)
            //    continue;
            if (c < 5)
                continue;
            for (size_t i = 0; i < indices[c].size(); ++i)
            {
                const auto color = colors[c % NUM_COLORS];

                auto idx = indices[c][i];
                auto& rect = boxes[c][idx];
                cv::Size s = frame.size();
                //if (rect.x < 0 || rect.y < 0 || rect.x + rect.width >= s.width || rect.y + rect.height >= s.height)
                //{
                //    std::cout << rect.x << "; " << rect.y << ":" << rect.width << ";" << rect.height << std::endl;
                //    std::cout << s.width << "; " << s.height << std::endl;
                //}
                if (rect.x < 0)
                {
                    rect.width += rect.x;
                    rect.x = 0;
                }
                if (rect.x + rect.width >= s.width)
                    rect.width = s.width - rect.x;
                if (rect.y < 0)
                {
                    rect.height += rect.y;
                    rect.y = 0;
                }
                if (rect.y + rect.height >= s.height)
                    rect.height = s.height - rect.y;
                cv::Rect roi(rect.x, rect.y, rect.width, rect.height);
                cv::Range rows(rect.x, rect.x + rect.width);
                cv::Range cols(rect.y, rect.y + rect.height);
                cv::Mat matRoi = frame(cols, rows);// rows, cols);
                cv::String name = groundtruthPath + "/";
                name += filenames[i].substr(filenames[i].find_last_of("/\\") + 1);// filenames[i];
                name = name.substr(0, name.size() - 4);
                while (cv::utils::fs::exists(name + ".png") == true)
                    name += "#";
                name += ".png";
                //cv::imwrite(cv::format("Results LP/Image%d.png", i), frame);
                cv::imwrite(name, matRoi);
                std::ostringstream label_ss;
                //label_ss << class_names[c] << ": " << std::fixed << std::setprecision(2) << scores[c][idx];
                label_ss << "license plate";

                auto label = label_ss.str();

                int baseline;
                auto label_bg_sz = cv::getTextSize(label.c_str(), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, 1, &baseline);

                cv::rectangle(frame, cv::Point(rect.x, rect.y), cv::Point(rect.x + rect.width, rect.y + rect.height), color, 3);

                if (rect.y - label_bg_sz.height - baseline < 10)
                {
                    //label.c_str()
                    cv::rectangle(frame, cv::Point(rect.x + rect.width - label_bg_sz.width, rect.y + rect.height - label_bg_sz.height - baseline - 10), cv::Point(rect.x + rect.width, rect.y + rect.height), color, cv::FILLED);
                    cv::putText(frame, label.c_str(), cv::Point(rect.x + rect.width - label_bg_sz.width, rect.y + rect.height - baseline - 5), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 0, 0));
                }
                else
                {
                    cv::rectangle(frame, cv::Point(rect.x, rect.y - label_bg_sz.height - baseline - 10), cv::Point(rect.x + label_bg_sz.width, rect.y), color, cv::FILLED);
                    cv::putText(frame, label.c_str(), cv::Point(rect.x, rect.y - baseline - 5), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 0, 0));
                }
                //auto rect = Render(frame, c, i, class_names, indices, boxes, scores, log_file);

                //if (class_names[c].compare("truck") != 0 && class_names[c].compare("bus") != 0 && class_names[c].compare("car") != 0)
                //    continue;
                //const auto color = colors[c % NUM_COLORS];

                //auto idx = indices[c][i];
                //const auto& rect = boxes[c][idx];
                //
                //std::ostringstream label_ss;
                //label_ss << class_names[c] << ": " << std::fixed << std::setprecision(2) << scores[c][idx];

                //auto label = label_ss.str();

                //cv::rectangle(frame, cv::Point(rect.x, rect.y), cv::Point(rect.x + rect.width, rect.y + rect.height), color, 3);


                //
                ////std::cout << label << std::endl;
                ////if (class_names[c].compare("truck") == 0)
                ////    std::cout << "Found a truck!" << std::endl;
                //int baseline;
                //auto label_bg_sz = cv::getTextSize(label.c_str(), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, 1, &baseline);
                //if (rect.y - label_bg_sz.height - baseline < 10)
                //{
                //    //label.c_str()
                //    cv::rectangle(frame, cv::Point(rect.x + rect.width - label_bg_sz.width, rect.y + rect.height - label_bg_sz.height - baseline - 10), cv::Point(rect.x + rect.width, rect.y + rect.height), color, cv::FILLED);
                //    cv::putText(frame, label.c_str(), cv::Point(rect.x + rect.width - label_bg_sz.width, rect.y + rect.height - baseline - 5), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 0, 0));
                //}
                //else
                //{
                //    cv::rectangle(frame, cv::Point(rect.x, rect.y - label_bg_sz.height - baseline - 10), cv::Point(rect.x + label_bg_sz.width, rect.y), color, cv::FILLED);
                //    cv::putText(frame, label.c_str(), cv::Point(rect.x, rect.y - baseline - 5), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 0, 0));
                //}


                //cv::dnn::blobFromImage(matRoi, blob_LP, 0.00392, cv::Size(608, 608), cv::Scalar(), true, false, CV_32F);
                //net_LP.setInput(blob_LP);
                //try
                //{
                //    net_LP.forward(detections_LP, output_names_LP);
                //}
                //catch (std::exception& ex)
                //{
                //    log_file << "We got a problem!: " << ex.what();
                //    break;
                //}
                //Yolo(matRoi, NUM_CLASSES_LP, CONFIDENCE_THRESHOLD, NMS_THRESHOLD, detections_LP, indices_LP, boxes_LP, scores_LP);
                //for (int cc = 0; cc < NUM_CLASSES_LP; ++cc)
                //{
                //    if (cc != 0)
                //        continue;
                //    for (size_t ii = 0; ii < indices_LP[cc].size(); ++ii)
                //    {
                //        Render(frame, cc, ii, class_names_LP, indices_LP, boxes_LP, scores_LP, log_file, rect.x, rect.y);
                //    }
                //}
            }
        }

        auto total_end = std::chrono::steady_clock::now();

        float inference_fps = 1000.0 / std::chrono::duration_cast<std::chrono::milliseconds>(dnn_end - dnn_start).count();
        float total_fps = 1000.0 / std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start).count();
        std::ostringstream stats_ss;
        stats_ss << std::fixed << std::setprecision(2);
        stats_ss << "Inference FPS: " << inference_fps << ", Total FPS: " << total_fps;
        auto stats = stats_ss.str();

        int baseline;
        auto stats_bg_sz = cv::getTextSize(stats.c_str(), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, 1, &baseline);
        cv::rectangle(frame, cv::Point(0, 0), cv::Point(stats_bg_sz.width, stats_bg_sz.height + 10), cv::Scalar(0, 0, 0), cv::FILLED);
        cv::putText(frame, stats.c_str(), cv::Point(0, stats_bg_sz.height + 5), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(255, 255, 255));

        cv::String name = outPath;
        name += "\\";
        name += filenames[i].substr(filenames[i].find_last_of("/\\") + 1);// filenames[i];
        name += ".png";
        //cv::imwrite(cv::format("Results LP/Image%d.png", i), frame);
        cv::imwrite(name, frame);
        //cv::namedWindow("output");
        //cv::imshow("output", frame);
        //cv::waitKey(60);
        //out.write(frame);
    }
    //source.release();
    //out.release();
    cv::destroyAllWindows();
    return 0;
}