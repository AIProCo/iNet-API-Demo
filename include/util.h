/*==============================================================================
* Copyright 2022 AIPro Inc.
* Author: Chun-Su Park (cspk@skku.edu)
=============================================================================*/
#ifndef UTIL_H
#define UTIL_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "global.h"

using namespace std;
using namespace cv;

const uchar colorTable[] = {
	169, 83,  255, 249, 249, 27,  127, 255, 212, 240, 255, 255, 245, 245, 220, 255, 228, 196, 255, 235, 205, 138,
	43,  226, 222, 184, 135, 95,  158, 160, 250, 235, 215, 210, 105, 30,  255, 127, 80,  100, 149, 237, 255, 248,
	220, 220, 20,  60,  0,   255, 255, 0,   139, 139, 184, 134, 11,  169, 169, 169, 189, 183, 107, 255, 140, 0,
	153, 50,  204, 233, 150, 122, 143, 188, 143, 0,   206, 209, 148, 0,   211, 255, 20,  147, 0,   191, 255, 30,
	144, 255, 178, 34,  34,  255, 250, 240, 34,  139, 34,  255, 0,   255, 220, 220, 220, 248, 248, 255, 255, 215,
	0,   218, 165, 32,  250, 128, 114, 210, 180, 140, 240, 255, 240, 255, 105, 180, 205, 92,  92,  255, 255, 240,
	240, 230, 140, 230, 230, 250, 255, 240, 245, 124, 252, 0,   255, 250, 205, 173, 216, 230, 240, 128, 128, 224,
	255, 255, 250, 250, 210, 211, 211, 211, 144, 238, 144, 255, 182, 193, 255, 160, 122, 32,  178, 170, 135, 206,
	250, 119, 136, 153, 119, 136, 153, 176, 196, 222, 255, 255, 224, 0,   255, 0,   50,  205, 50,  250, 240, 230,
	255, 0,   255, 102, 205, 170, 186, 85,  211, 147, 112, 219, 60,  179, 113, 123, 104, 238, 0,   250, 154, 72,
	209, 204, 199, 21,  133, 245, 255, 250, 255, 228, 225, 255, 228, 181, 255, 222, 173, 253, 245, 230, 128, 128,
	0,   107, 142, 35,  255, 165, 0,   255, 69,  0,   218, 112, 214, 238, 232, 170, 152, 251, 152, 175, 238, 238,
	219, 112, 147, 255, 239, 213, 255, 218, 185, 205, 133, 63,  255, 192, 203, 221, 160, 221, 176, 224, 230, 128,
	0,   128, 255, 0,   0,   188, 143, 143, 65,  105, 225, 139, 69,  19,  0,   128, 0,   240, 248, 255 };

/// Data structure to represent Video Caputer, Writer
struct VideoCW {
	VideoCapture* videoCapturer;
	VideoWriter* videoWriter;
	string inPath;
	string outPath;
	int frameWidth;
	int frameHeight;
	float fps;

	VideoCapture& cap() const {
		return *(videoCapturer);
	}
	VideoWriter& writer() const {
		return *(videoWriter);
	}

	friend VideoCW& operator>>(VideoCW& videoCW, Mat& frame) {
		videoCW.cap() >> frame;
		return videoCW;
	}

	friend VideoCW& operator<<(VideoCW& videoCW, Mat& frame) {
		videoCW.writer().write(frame);
		return videoCW;
	}
};

/// this is a helper class to read video files in a certain directory
class VideoDir {
private:
	vector<string> inPaths;
	vector<string> outPaths;
	vector<VideoCW*> videoCWs;
	vector<int> frameWidths;
	vector<int> frameHeights;
	vector<float> fpss;

public:
	VideoCW& operator[](int idx) const {
		return *videoCWs[idx];
	}

	VideoDir() {
	}

	VideoDir(const vector<string>& inFilepaths, const vector<string>& outFilepaths) {
		init(inFilepaths, outFilepaths);
	}

	void init(const vector<string>& inFilepaths, const vector<string>& outFilepaths) {
		setVideoCWFiles(inFilepaths, outFilepaths);
		setFrameWidthsHeights(videoCWs);
	}

	virtual ~VideoDir() {
		freeVideoCWFiles(this->videoCWs);
	}

	int size() {
		return videoCWs.size();
	}

	vector<int> getFrameWidths() {
		return frameWidths;
	}

	vector<int> getFrameHeights() {
		return frameHeights;
	}

	vector<float> getFpss() {
		return fpss;
	}

	void setVideoCWFiles(const vector<string>& inFilepaths, const vector<string>& outFilepaths) {
		for (size_t i = 0; i < inFilepaths.size(); i++) {
			string inFilepath = inFilepaths[i];
			string outFilepath = outFilepaths[i];

			VideoCW* videoCW;
			VideoCapture* capturer = new VideoCapture(inFilepath);

			if (!capturer->isOpened()) {
				cout << "Error opening video stream" << inFilepath << endl;
				break;
			}

			videoCW = new VideoCW();

			if (videoCW == NULL) {
				cout << "videoCW memory allocation error!!" << endl;
				break;
			}

			videoCW->videoCapturer = capturer;
			videoCW->frameWidth = capturer->get(CAP_PROP_FRAME_WIDTH);
			videoCW->frameHeight = capturer->get(CAP_PROP_FRAME_HEIGHT);
			videoCW->fps = capturer->get(CAP_PROP_FPS);
			videoCW->videoWriter = new VideoWriter(outFilepath, VideoWriter::fourcc('m', 'p', '4', 'v'), videoCW->fps,
				Size(videoCW->frameWidth, videoCW->frameHeight));  ///*.mp4 format
			videoCWs.push_back(videoCW);
		}
	}

	static void freeVideoCWFiles(vector<VideoCW*>& videoCWs) {
		for (int i = 0; i < videoCWs.size(); i++) {
			videoCWs[i]->videoCapturer->release();
			videoCWs[i]->videoWriter->release();
			delete videoCWs[i]->videoCapturer;
			delete videoCWs[i]->videoWriter;
			delete videoCWs[i];
		}
	}

	void setFrameWidthsHeights(vector<VideoCW*>& videoCWs) {
		for (int i = 0; i < videoCWs.size(); i++) {
			frameWidths.push_back(videoCWs[i]->frameWidth);
			frameHeights.push_back(videoCWs[i]->frameHeight);
			fpss.push_back(videoCWs[i]->fps);
		}
	}
};

/// @brief  A utility function to print out the elements of vector
template <typename T>
std::ostream& operator<<(std::ostream& os, std::vector<T> vec) {
	os << "{ ";
	std::copy(vec.begin(), vec.end(), std::ostream_iterator<T>(os, " "));
	os << "}";
	return os;
}

/// @brief an utility class to visualize the results
class Vis {
public:
	/// draw bboxes with text info for each detected object
	static void drawBoxes(Mat& matImg, vector<Rect> boxes, vector<Scalar> boxColors, vector<vector<string>> texts,
		Scalar textColor = Scalar(0, 0, 0), int fontFace = FONT_HERSHEY_SIMPLEX,
		double fontScale = 0.5f, int thickness = 1, int vSpace = 4, int hSpace = 6,
		int textBlockTopOffset = 0) {
		for (int i = 0; i < boxes.size(); i++) {
			Rect box = boxes[i];
			int l = box.x;
			int t = box.y;
			int r = box.x + box.width;
			int b = box.y + box.height;

			/// draw bboxes
			rectangle(matImg, Point(l, t), Point(r, b), boxColors[i], thickness + 1);

			/// calculate boxs
			Size bboxTexts = getBoxForTexts(texts[i], fontFace, fontScale, thickness, vSpace, hSpace);

			/// filled boxes (to show text info));
			Point topLeftBox = Point(r, t + textBlockTopOffset);
			Point rightBottomBox = Point(r + bboxTexts.width, t + bboxTexts.height + textBlockTopOffset);
			rectangle(matImg, topLeftBox, rightBottomBox, boxColors[i], FILLED);

			/// draw texts
			drawTexts(matImg, topLeftBox, texts[i], textColor, fontFace, fontScale, thickness, vSpace, hSpace);
		}
	}

	/// draw text block, i.e, texts withing a bounding rectangle.
	/// e.g: show tracking information, gender counting
	static void drawTextBlock(Mat& matImg, Point topLeftBox, vector<string> texts, double fontScale = 0.5f,
		int thickness = 1, Scalar boxColor = Scalar(0, 255, 0),
		Scalar textColor = Scalar(0, 255, 0), int fontFace = FONT_HERSHEY_SIMPLEX,
		int vSpace = 10, int hSpace = 10) {
		Size txtSize = getBoxForTexts(texts, fontFace, fontScale, thickness, vSpace, hSpace);
		Point rightBottom = Point(topLeftBox.x + txtSize.width, topLeftBox.y + txtSize.height);
		/// draw bboxes
		rectangle(matImg, topLeftBox, rightBottom, boxColor, thickness + 1);

		/// draw text
		drawTexts(matImg, topLeftBox, texts, textColor, fontFace, fontScale, thickness, vSpace, hSpace);
	}

	/// draw texts in multiple lines
	/// draw text only (no bouding box)
	static void drawTexts(Mat& matImg, Point startPoint, vector<string> texts, Scalar textColor = Scalar(0, 0, 0),
		int fontFace = FONT_HERSHEY_SIMPLEX, double fontScale = 0.5f, int thickness = 1,
		int vSpace = 10, int hSpace = 10) {
		int baseLine = 0;
		for (size_t i = 0; i < texts.size(); i++) {
			Size lineSize = getTextSize(texts[i], fontFace, fontScale, thickness, &baseLine);
			int xText = startPoint.x + hSpace;
			int yText = startPoint.y + ((i + 1) * (vSpace + lineSize.height));
			putText(matImg, texts[i], Point2f(xText, yText), fontFace, fontScale, textColor, thickness);
		}
	}

	/// func to calculate the size of bounding box for multiple texts
	static Size getBoxForTexts(vector<string> texts, int fontFace = FONT_HERSHEY_SIMPLEX, double fontScale = 0.5f,
		int thickness = 1, int vSpace = 10, int hSpace = 10) {
		int width = 0;
		int height = 0;
		int baseLine = 0;

		for (size_t i = 0; i < texts.size(); i++) {
			Size lineSize = getTextSize(texts[i], fontFace, fontScale, thickness, &baseLine);
			width = max(width, lineSize.width);
			height += lineSize.height + vSpace;
		}
		width += (2 * hSpace);
		height += vSpace;

		return Size(width, height);
	}
};

#endif  // UTIL_H