// FacialDetection.cpp : This file contains the 'main' function. Program execution begins and ends there.
//


#include <iostream>

#include "opencv2/objdetect.hpp"
//#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/video/background_segm.hpp"

using namespace std;
using namespace cv;


int main(int argc, char** argv)
{
	CascadeClassifier face_cascade, eyes_cascade, mouth_cascade, nose_cascade, ears_cascade;
	String face_cascade_name = "haarcascade_frontalface_alt.xml";
	String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
	String mouth_cascade_name = "haarcascade_mcs_mouth-1.xml";
	String nose_cascade_name = "haarcascade_mcs_nose-1.xml";
	std::vector<Rect> faces;
	cv::Mat frame;
	cv::Mat frame_gray;


	// Open the default camera
	cv::VideoCapture cap(0);
	if (!cap.isOpened())
	{
		std::cerr << "Unable to open the camera\n";
		return -1;
	}

	// Create a window to display the video
	//cv::namedWindow("Webcam Feed", cv::WINDOW_NORMAL);

	//-- 1. Load the cascades
	if (!face_cascade.load(face_cascade_name))
	{
		std::cout << "Error loading face cascade\n";
		return 0;
	}

	if (!eyes_cascade.load(eyes_cascade_name))
	{
		std::cout << "Error loading eyes cascade\n";
		return 0;
	}

	if (!mouth_cascade.load(mouth_cascade_name))
	{
		std::cout << "Error loading mouth cascade\n";
		return 0;
	}

	if (!nose_cascade.load(nose_cascade_name))
	{
		std::cout << "Error loading nose cascade\n";
		return 0;
	}

	// Capture frames continuously and display them in the window
	while (true)
	{
		cap >> frame;

		// Check if the frame is empty
		if (frame.empty())
		{
			std::cerr << "Unable to capture frame\n";
			break;
		}

		// Display the frame
		cv::imshow("Webcam Source Feed", frame);

		cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
		equalizeHist(frame_gray, frame_gray);

		//-- Detect faces
		face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

		for (size_t i = 0; i < faces.size(); i++)
		{
			//	Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);
			//	ellipse(frame, center, Size(faces[i].width / 2, faces[i].height / 2), 0, 0, 360, Scalar(255, 0, 255), 4, 8, 0);

			//-- In each face, draw a rectangle
			Point pt1(faces[i].x, faces[i].y);
			Point pt2(faces[i].x + faces[i].width, faces[i].y + faces[i].height);
			rectangle(frame, pt1, pt2, Scalar(255, 0, 255), 4, 8, 0);

			// finding region of interest
			Mat faceROI = frame_gray(faces[i]);

			//-- In each face ROI, detect eyes
			std::vector<Rect> eyes;
			eyes_cascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));
			for (size_t j = 0; j < eyes.size(); j++)
			{
				Point eye_center(faces[i].x + eyes[j].x + eyes[j].width / 2, faces[i].y + eyes[j].y + eyes[j].height / 2);
				int radius = cvRound((eyes[j].width + eyes[j].height) * 0.25);
				circle(frame, eye_center, radius, Scalar(255, 0, 0), 4, 8, 0);
			}
/*
			//-- In each face ROI, detect mouth
			std::vector<Rect> mouth;
			mouth_cascade.detectMultiScale(faceROI, mouth, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));
			for (size_t k = 0; k < mouth.size(); k++)
			{
				Point mouth_center(faces[i].x + mouth[k].x + mouth[k].width / 2, faces[i].y + mouth[k].y + mouth[k].height / 2);
				ellipse(frame, mouth_center, Size(mouth[k].width / 2, mouth[k].height / 2), 0, 0, 360, Scalar(0, 255, 0), 4, 8, 0);
			}


			//-- In each face ROI, detect nose
			std::vector<Rect> nose;
			nose_cascade.detectMultiScale(faceROI, nose, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));
			for (size_t l = 0; l < nose.size(); l++)
			{
				//-- In each nose, draw a rectangle
				Point pt3(nose[l].x, nose[l].y);
				Point pt4(nose[l].x + nose[l].width, nose[l].y + nose[l].height);
				rectangle(frame, pt3, pt4, Scalar(255, 255, 255), 4, 8, 0);

			}
*/

		}

		cv::imshow("Webcam Dest Feed", frame);

		// Check if the user pressed the 'Esc' key
		if (cv::waitKey(1) == 27)
		{
			break;
		}

	}

	// Release the camera and destroy the window
	cap.release();
	cv::destroyAllWindows();

	return 0;
}

