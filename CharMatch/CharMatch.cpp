// CharMatch.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include <opencv2/opencv.hpp>
//#include <opencv2/core.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/imgcodecs.hpp>

#include <iostream>

const int MIN_CONTOUR_AREA = 100;
const int RESIZED_IMAGE_WIDTH = 20;
const int RESIZED_IMAGE_HEIGHT = 30;

using namespace cv;
//using namespace cv::ml;

int main(int, char**)
{
    // Load the classifier
    Ptr<cv::ml::KNearest> knn = Algorithm::load<cv::ml::KNearest>("knn.xml", "KNearest");

    //std::cout << "Number of classes: " << knn->getClassCount() << std::endl;
    //std::cout << "Number of samples: " << knn->getSampleCount() << std::endl;
    std::cout << "Algorithm Type: " << knn->getAlgorithmType() << std::endl;
    std::cout << "Number of features: " << knn->getVarCount() << std::endl;
    std::cout << "Default K: " << knn->getDefaultK() << std::endl;

    // Load an image
    Mat img = imread("image-A.png");
    if (img.empty()) {                               // if unable to open image
        std::cout << "error: image not read from file\n\n";         // show error message on command line
        return 0;                                                  // and exit program
    }

    Mat imgGrayscale;
    Mat imgBlurred;
    Mat imgThresh;
    Mat imgThreshCopy;
    Mat imgResized;

    std::vector<std::vector<cv::Point> > ptContours;        // declare contours vector
    std::vector<cv::Vec4i> v4iHierarchy;                    // declare contours hierarchy

 
    cv::cvtColor(img, imgGrayscale, COLOR_BGR2GRAY);        // convert to grayscale
    
  /*
    cv::GaussianBlur(imgGrayscale,              // input image
        imgBlurred,                             // output image
        cv::Size(5, 5),                         // smoothing window width and height in pixels
        0);                                     // sigma value, determines how much the image will be blurred, zero makes function choose the sigma value

                                                // filter image from grayscale to black and white
    cv::adaptiveThreshold(imgBlurred,           // input image
        imgThresh,                              // output image
        255,                                    // make pixels that pass the threshold full white
        cv::ADAPTIVE_THRESH_GAUSSIAN_C,         // use gaussian rather than mean, seems to give better results
        cv::THRESH_BINARY_INV,                  // invert so foreground will be white, background will be black
        11,                                     // size of a pixel neighborhood used to calculate threshold value
        2);                                     // constant subtracted from the mean or weighted mean

    //cv::imshow("imgThresh", imgThresh);         // show threshold image for reference

    imgThreshCopy = imgThresh.clone();          // make a copy of the thresh image, this in necessary b/c findContours modifies the image

    cv::findContours(imgThreshCopy,             // input image, make sure to use a copy since the function will modify this image in the course of finding contours
        ptContours,                             // output contours
        v4iHierarchy,                           // output hierarchy
        cv::RETR_EXTERNAL,                      // retrieve the outermost contours only
        cv::CHAIN_APPROX_SIMPLE);               // compress horizontal, vertical, and diagonal segments and leave only their end points
*/


/*
    for (int i = 0; i < ptContours.size(); i++) {                           // for each contour
        if (cv::contourArea(ptContours[i]) > MIN_CONTOUR_AREA) {                // if contour is big enough to consider
            cv::Rect boundingRect = cv::boundingRect(ptContours[i]);                // get the bounding rect

            cv::Mat matROI = imgThresh(boundingRect);           // get ROI image of bounding rect
          
            cv::resize(matROI, imgResized, cv::Size(RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT));     // resize image, this will be more consistent for recognition and storage

            //std::cout << "Resized: " << imgResized << std::endl << std::endl;

            Mat reshaped_img = imgResized.reshape(1, 1);
            //std::cout << "Reshaped: " << reshaped_img << std::endl << std::endl;

            int intChar = cv::waitKey(0);           // get key press

            if (intChar == 27) {        // if esc key was pressed
                return 0;              // exit program
            }

            // predict class for a new sample
            //int response = knn->predict(reshaped_img);
            //std::cout << "Alphabet label: " << char(response) << std::endl;

            Mat results;
            Mat neighborResponses;
            Mat dists;
            knn->findNearest(reshaped_img, knn->getDefaultK(), results, neighborResponses, dists);

            // Output the nearest neighbors and their distances
            std::cout << "Nearest neighbors: " << neighborResponses << std::endl;
            std::cout << "Distances: " << dists << std::endl;

        }
    }
*/



    return 0;
}