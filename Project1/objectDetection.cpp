#include "opencv2/objdetect.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/face.hpp"

#include <iostream>
#include <stdio.h>

// Haar cascade locations
cv::String face_cascade_name = "haar/haarcascade_frontalface_alt.xml";
cv::String mouth_cascade_name = "haar/haarcascade_mcs_mouth.xml";

// Extra pictures location
cv::String m_1 = "mPic/1.jpeg";
cv::String m_2 = "mPic/3.jpg";

// The name of people in the image
cv::String auth1 = "David";
cv::String auth2 = "Zsolti";

// The name of the window
cv::String window_name = "Face Recognition";

// Declare cascades
cv::CascadeClassifier face_cascade;
cv::CascadeClassifier mouth_cascade;

// Get which person is on the screen based on prediction
cv::String facedetector(int prediction) {
	if (prediction == 0)
		return auth1;
	else if (prediction == 1)
		return auth2;
	return "Unknown";
}

int main(void){

	// Declare vectors
	std::vector<cv::Mat> images;
	std::vector<int> labels;

	// Get the pictures of the first person
	cv::String path("database/1/*.jpg");
	std::vector<cv::String> fn;
	cv::glob(path, fn, true);
	for (unsigned k = 0; k<fn.size(); ++k){
		cv::Mat im = cv::imread(fn[k],0);
		if (im.empty()) continue;
		images.push_back(im);
		labels.push_back(0);
	}

	// Get the pictures of the second person
	cv::String path2("database/2/*.jpg");
	std::vector<cv::String> fn2;
	cv::glob(path2, fn2, true);
	for (unsigned k = 0; k<fn2.size(); ++k){
		cv::Mat im = cv::imread(fn2[k],0);
		if (im.empty()) continue;
		images.push_back(im);
		labels.push_back(1);
	}

	// Declare everything we will use later on
	cv::Mat frame;

	if (!face_cascade.load(face_cascade_name)) {
		printf("--(!)Error loading face cascade\n");
		return -1; 
	};

	if (!mouth_cascade.load(mouth_cascade_name)) {
		printf("--(!)Error loading mouth cascade\n");
		return -1;
	};

	int im_width = images[0].cols;
	int im_height = images[0].rows;

	// Resize images for the training process
	for (unsigned j = 0; j < images.size(); j++) {
		cv::Size size(im_width, im_height);
		cv::resize(images[j], images[j], size);
	}

	// Create our face recognizer based on Fisher Face Recognition
	cv::Ptr<cv::face::FaceRecognizer> model = cv::face::createFisherFaceRecognizer();

	// Train the recognizer
	model->train(images, labels);

	// Open the web cam
	cv::VideoCapture cap(0);
	if (!cap.isOpened()) {
		std::cerr << "Capture cannot be opened." << std::endl;
		return -1;
	}

	for (;;) {

		// Capture current image
		cap >> frame;

		// Set fps for a little performance boost
		cap.set(CV_CAP_PROP_FPS, 25);

		// Clone current image
		cv::Mat original = frame.clone();

		// Store gray image
		cv::Mat gray;
		cv::cvtColor(original, gray, CV_BGR2GRAY);

		// Vector to store every faces location on the screen
		std::vector< cv::Rect_<int> > faces;

		// Get faces
		face_cascade.detectMultiScale(gray, faces);

		// Inspect and modify every single face
		for (unsigned i = 0; i < faces.size(); i++) {

			// Select only closer faces (Fix for fake faces in the background)
			if (faces[i].height >= 100 && faces[i].width >= 100) {

				// Get current face coordinates
				cv::Rect face_i = faces[i];

				// Get the picture from gray with the given locations (The current face)
				cv::Mat face = gray(face_i);

				// Resize current face
				cv::Mat face_resized;
				cv::resize(face, face_resized, cv::Size(im_width, im_height), 1.0, 1.0, CV_INTER_CUBIC);

				// Get the prediction from our trained recognizer
				int prediction = model->predict(face_resized);

				// Visual help for showing the face
				cv::rectangle(original, face_i, CV_RGB(0, 255, 0), 1);
				std::string box_text = facedetector(prediction);
				int pos_x = std::max(face_i.tl().x - 10, 0);
				int pos_y = std::max(face_i.tl().y - 10, 0);
				cv::putText(original, box_text, cv::Point(pos_x, pos_y), CV_FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0, 255, 0), 2.0);

				// Get the mouth of the face
				std::vector< cv::Rect_<int> > mouths;
				cv::Rect lowerFaceCord = cv::Rect(face_i.x,face_i.y+2*face_i.height/3,face_i.width,face_i.height/3);
				cv::Mat lowerFace = gray(lowerFaceCord);
				mouth_cascade.detectMultiScale(lowerFace, mouths);
				if (mouths.size() > 0) {
					cv::Rect mouth_0 = mouths[0];
					cv::Rect gMouth = cv::Rect(lowerFaceCord.x + mouth_0.x, lowerFaceCord.y + mouth_0.y,mouth_0.width,mouth_0.height);
					if (gMouth.width >= lowerFaceCord.width / 4) {
						cv::Mat img, img2;

						// Get the perfect extra stuff for our current person
						if (prediction == 0) {
							img = cv::imread(m_1);
							if (img.data) {
								cv::resize(img, img2, cv::Size(gMouth.width*2, gMouth.height*2), 1.0, 1.0, CV_INTER_CUBIC);
								for (int i = 0; i < img2.rows; i++) {
									for (int j = 0; j < img2.cols; j++) {
										if (img2.at<cv::Vec3b>(i, j)[0] < 225 &&
											img2.at<cv::Vec3b>(i, j)[1] < 225 &&
											img2.at<cv::Vec3b>(i, j)[2] < 225) {
											if ((i + gMouth.y - gMouth.height) < original.size().height &&
												(j + gMouth.x - gMouth.width / 2) < original.size().width) {
												original.at<cv::Vec3b>(i + gMouth.y - gMouth.height, j + gMouth.x - gMouth.width / 2)[0] = 0;
												original.at<cv::Vec3b>(i + gMouth.y - gMouth.height, j + gMouth.x - gMouth.width / 2)[1] = 0;
												original.at<cv::Vec3b>(i + gMouth.y - gMouth.height, j + gMouth.x - gMouth.width / 2)[2] = 0;
											}
										}
									}
								}
							}
						}
						else {
							img = cv::imread(m_2);
							if (img.data) {
								cv::resize(img, img2, cv::Size(gMouth.width, gMouth.height*3), 1.0, 1.0, CV_INTER_CUBIC);
								for (int i = 0; i < img2.rows; i++) {
									for (int j = 0; j < img2.cols; j++) {
										if (img2.at<cv::Vec3b>(i, j)[0] < 225 &&
											img2.at<cv::Vec3b>(i, j)[1] < 225 &&
											img2.at<cv::Vec3b>(i, j)[2] < 225) {
											if ((i + gMouth.y + gMouth.height/2) < original.size().height &&
												j + gMouth.x < original.size().width) {
												original.at<cv::Vec3b>(i + gMouth.y + gMouth.height/2, j + gMouth.x)[0] = 0;
												original.at<cv::Vec3b>(i + gMouth.y + gMouth.height/2, j + gMouth.x)[1] = 0;
												original.at<cv::Vec3b>(i + gMouth.y + gMouth.height/2, j + gMouth.x)[2] = 0;
											}
										}
									}
								}
							}
						}
					}
				}
			}
		}

		// Show the current picture
		cv::imshow("face_recognizer", original);

		// Esc exit
		char key = (char)cv::waitKey(20);
		if (key == 27)
			break;
	}

	return 0;
}