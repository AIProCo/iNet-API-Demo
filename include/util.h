/*==============================================================================
* Copyright 2024 AIPro Inc.
* Author: Chun-Su Park (cspk@skku.edu)
=============================================================================*/
#ifndef UTIL_H
#define UTIL_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <filesystem>
#include <unordered_map>

// for sleep
#include <thread>
#include <chrono>

#include "global.h"

#ifdef _WIN32
#include <Windows.h>
#else
// for get executable path
#include <unistd.h>
#include <limits.h>
#endif

using namespace std;
using namespace cv;
namespace fs = std::filesystem;
using TimePoint = std::chrono::steady_clock::time_point;

/// @brief  A utility function to print out the elements of vector
template <typename T>
std::ostream &operator<<(std::ostream &os, std::vector<T> vec) {
    os << "{ ";
    std::copy(vec.begin(), vec.end(), std::ostream_iterator<T>(os, " "));
    os << "}";
    return os;
}
/// @brief  A utility function to simulate sleep function
inline void universal_sleep(long long ms) {
#ifdef _WIN32
    Sleep(ms);
#else
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
#endif
}

/// @brief  A utility class to work with files and directories
class FileUtil {
   public:
    // get Home directory of current user in Linux
    static std::string getHomeDirPath() {
        return fs::path(getenv("HOME")).string();
    }

    // get current working directory of executable file
    static std::string getExecDirPath() {
#ifdef _WIN32
        char buffer[MAX_PATH];
        GetModuleFileNameA(NULL, buffer, MAX_PATH);
        fs::path p(buffer);
        return p.parent_path().string();
#else
        char result[PATH_MAX];
        ssize_t count = readlink("/proc/self/exe", result, PATH_MAX);
        return fs::path(std::string(result, (count > 0) ? count : 0)).parent_path().string();
#endif
    }

    static std::string universalNormPath(const std::string &relativePath, fs::path baseDir = "") {
#ifdef _WIN32
        // not needed, just return relativePath
        return relativePath;
#else
        // for Linux; combine baseDir and relativePath to get absolute path
        fs::path combinedPath = baseDir / relativePath;
        fs::path absolutePath = fs::absolute(combinedPath);
        return fs::weakly_canonical(absolutePath).string();
#endif
    }
};

/// @brief an utility class to visualize the results
class Vis {
   public:
    /// draw bboxes with text info for each detected object
    static void drawBoxes(Mat &matImg, vector<Rect> boxes, vector<Scalar> boxColors, vector<vector<string>> texts,
                          vector<bool> emphasizes, Scalar textColor = Scalar(0, 0, 0),
                          int fontFace = FONT_HERSHEY_SIMPLEX, double fontScale = 0.5f, int thickness = 1,
                          int vSpace = 4, int hSpace = 6, int textBlockTopOffset = 0) {
        for (int i = 0; i < boxes.size(); i++) {
            Rect box = boxes[i];
            int l = box.x;
            int t = box.y;
            int r = box.x + box.width;
            int b = box.y + box.height;

            /// draw bboxes
            if (emphasizes[i])
                rectangle(matImg, Point(l, t), Point(r, b), boxColors[i], thickness + 10);
            else
                rectangle(matImg, Point(l, t), Point(r, b), boxColors[i], thickness + 1);

            if (texts[i].size() > 0) {
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
    }

    static void drawBoxesFD(Mat &matImg, vector<Rect> boxes, vector<Scalar> boxColors, vector<vector<string>> texts,
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
            rectangle(matImg, Point(l, t), Point(r, b), boxColors[i], thickness + 5);

            if (texts[i].size() > 0) {
                /// calculate boxs
                Size bboxTexts = getBoxForTexts(texts[i], fontFace, fontScale, thickness, vSpace, hSpace);

                /// filled boxes (to show text info));
                Point topLeftBox = Point(l, t + textBlockTopOffset);
                Point rightBottomBox = Point(l + bboxTexts.width, t + bboxTexts.height + textBlockTopOffset);
                rectangle(matImg, topLeftBox, rightBottomBox, boxColors[i], FILLED);

                /// draw texts
                drawTexts(matImg, topLeftBox, texts[i], textColor, fontFace, fontScale, thickness, vSpace, hSpace);
            }
        }
    }
    /// draw text block, i.e, texts withing a bounding rectangle.
    /// e.g: show tracking information, gender counting
    static void drawTextBlock(Mat &matImg, Point topLeftBox, vector<string> texts, double fontScale = 0.5f,
                              int thickness = 1, Scalar boxColor = Scalar(255, 255, 255),
                              Scalar textColor = Scalar(255, 255, 255), int fontFace = FONT_HERSHEY_SIMPLEX,
                              int vSpace = 10, int hSpace = 10) {
        Size txtSize = getBoxForTexts(texts, fontFace, fontScale, thickness, vSpace, hSpace);
        Point rightBottom = Point(topLeftBox.x + txtSize.width, topLeftBox.y + txtSize.height);

        if (txtSize.width >= matImg.cols || txtSize.height >= matImg.rows)
            return;

        if (rightBottom.x > matImg.cols) {
            int shiftX = rightBottom.x - matImg.cols;
            topLeftBox.x -= shiftX;
            rightBottom.x -= shiftX;
        }

        if (rightBottom.y > matImg.rows) {
            int shiftY = rightBottom.y - matImg.rows;
            topLeftBox.y -= shiftY;
            rightBottom.y -= shiftY;
        }

        /// draw canvas
        Mat textRegion = matImg(Rect(topLeftBox, rightBottom));
        textRegion -= Scalar(100, 100, 100);

        /// draw bboxes
        rectangle(matImg, topLeftBox, rightBottom, boxColor, thickness + 1);

        /// draw text
        drawTexts(matImg, topLeftBox, texts, textColor, fontFace, fontScale, thickness, vSpace, hSpace);
    }

    /// e.g: show tracking information, gender counting
    static void drawTextBlock2(Mat &matImg, Point topLeftBox, vector<string> texts, double fontScale = 0.2f,
                               int thickness = 1, Scalar boxColor = Scalar(45, 45, 255),
                               Scalar textColor = Scalar(255, 255, 255), int fontFace = FONT_HERSHEY_SIMPLEX,
                               int vSpace = 10, int hSpace = 10) {
        Size txtSize = getBoxForTexts(texts, fontFace, fontScale, thickness, vSpace, hSpace);
        Point rightBottom = Point(topLeftBox.x + txtSize.width, topLeftBox.y + txtSize.height);

        if (txtSize.width >= matImg.cols || txtSize.height >= matImg.rows)
            return;

        if (rightBottom.x > matImg.cols) {
            int shiftX = rightBottom.x - matImg.cols;
            topLeftBox.x -= shiftX;
            rightBottom.x -= shiftX;
        }

        if (rightBottom.y > matImg.rows) {
            int shiftY = rightBottom.y - matImg.rows;
            topLeftBox.y -= shiftY;
            rightBottom.y -= shiftY;
        }

        /// draw canvas
        Mat textRegion = matImg(Rect(topLeftBox, rightBottom));
        textRegion -= Scalar(150, 150, 150);

        /// draw bboxes
        rectangle(matImg, topLeftBox, rightBottom, boxColor, 0);

        /// draw text
        drawTexts(matImg, topLeftBox, texts, textColor, fontFace, fontScale, thickness, vSpace, hSpace);
    }

    /// draw text block, i.e, texts withing a bounding rectangle.
    /// e.g: show tracking information, gender counting
    static void drawTextBlockFD(Mat &matImg, FDRecord &fdRcd, int vchID, int top, string text, double fontScale = 0.5f,
                                int thickness = 1, Scalar boxColor = Scalar(255, 255, 255),
                                Scalar textColor = Scalar(255, 255, 255), int fontFace = FONT_HERSHEY_SIMPLEX,
                                int vSpace = 10, int hSpace = 10) {
        vector<string> texts;
        texts.push_back(text);

        // Size txtSize = getBoxForTexts(texts, fontFace, fontScale, thickness, vSpace, hSpace);
        Size txtSize = getTextSize(text, fontFace, fontScale, thickness, 0);
        txtSize.height += 2 * vSpace;

        Point topLeftBox = Point(matImg.cols - 540, top);
        Point rightBottom = Point(topLeftBox.x + 520, topLeftBox.y + txtSize.height);

        /// draw canvas
        Mat textRegion = matImg(Rect(topLeftBox, rightBottom));
        textRegion -= Scalar(100, 100, 100);

        /// draw bboxes
        rectangle(matImg, topLeftBox, rightBottom, boxColor, thickness + 1);

        /// draw text
        int xText = topLeftBox.x + hSpace;
        int yText = topLeftBox.y + txtSize.height - vSpace;
        putText(matImg, text, Point2f(xText, yText), fontFace, fontScale, textColor, thickness);

        /// Draw Graph
        Point topLeftGraph = Point(topLeftBox.x, rightBottom.y);
        Point rightBottomGraph = Point(rightBottom.x, rightBottom.y + 2 * txtSize.height);

        /// draw canvas
        Mat graphRegion = matImg(Rect(topLeftGraph, rightBottomGraph));
        graphRegion -= Scalar(100, 100, 100);

        /// draw bboxes
        rectangle(matImg, topLeftGraph, rightBottomGraph, boxColor, thickness + 1);

        Point g(10, 10);
        Rect insideRect = Rect(topLeftGraph + g, rightBottomGraph - g);
        Mat insideRegion = matImg(insideRect);

        insideRegion -= Scalar(100, 100, 100);

        putText(matImg, String("prob"), Point(insideRect.x, insideRect.y + insideRect.height / 2), FONT_HERSHEY_PLAIN,
                fontScale, textColor, thickness);
        putText(matImg, String("1"), Point(insideRect.x, insideRect.y + 12), FONT_HERSHEY_PLAIN, fontScale, textColor,
                thickness);
        putText(matImg, String("0"), Point(insideRect.x, insideRect.y + insideRect.height - 5), FONT_HERSHEY_PLAIN,
                fontScale, textColor, thickness);

        vector<Point> firePts, smokePts;
        deque<float> &fireProbs = fdRcd.fireProbsMul[vchID];
        deque<float> &smokeProbs = fdRcd.smokeProbsMul[vchID];

        int windowSize = fireProbs.size();
        float deltaX = (float)insideRect.width / (windowSize - 1);
        float deltaY = (float)insideRect.height;

        for (int i = 0; i < windowSize; i++) {
            firePts.push_back(Point(insideRect.x + i * deltaX, insideRect.y + (1.0f - fireProbs[i]) * deltaY));
            smokePts.push_back(Point(insideRect.x + i * deltaX, insideRect.y + (1.0f - smokeProbs[i]) * deltaY));
        }

        polylines(matImg, firePts, false, Scalar(0, 0, 255), 2);
        polylines(matImg, smokePts, false, Scalar(220, 200, 200), 2);
    }

    /// draw texts in multiple lines
    /// draw text only (no bouding box)
    static void drawTexts(Mat &matImg, Point startPoint, vector<string> texts, Scalar textColor = Scalar(0, 0, 0),
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

/// @brief an utility class to print out colored text base on the code at https://stackoverflow.com/a/67195569. See
/// linux ANSI color codes: https://stackoverflow.com/a/45300654 for deeper understanding.
// How to use: PPrint::print("Hello World", "red", "yellow"); // print red text on yellow background

// !IMPORTANT: to use colored text, you need to define _COLORED_LOG as preprocessor definition in your project. See
// `docs\[hahv]_setup_preprocessor definition.md` for more details.
class PPrint {
   public:
    static unordered_map<string, int> colorToIntMap() {
        return {
            {"black", 0},  {"dark_blue", 1},  {"dark_green", 2}, {"light_blue", 3}, {"dark_red", 4}, {"magenta", 5},
            {"orange", 6}, {"light_gray", 7}, {"gray", 8},       {"blue", 9},       {"green", 10},   {"cyan", 11},
            {"red", 12},   {"pink", 13},      {"yellow", 14},    {"white", 15}  // default
        };
    }

    static unordered_map<string, string> textColorMap() {
        return {{"black", "30"},    {"dark_blue", "34"}, {"dark_green", "32"}, {"light_blue", "36"},
                {"dark_red", "31"}, {"magenta", "35"},   {"orange", "33"},     {"light_gray", "37"},
                {"gray", "90"},     {"blue", "94"},      {"green", "92"},      {"cyan", "96"},
                {"red", "91"},      {"pink", "95"},      {"yellow", "93"},     {"white", "97"}};
    }

    static unordered_map<string, string> bgColorMap() {
        return {{"black", "40"},    {"dark_blue", "44"}, {"dark_green", "42"}, {"light_blue", "46"},
                {"dark_red", "41"}, {"magenta", "45"},   {"orange", "43"},     {"light_gray", "47"},
                {"gray", "100"},    {"blue", "104"},     {"green", "102"},     {"cyan", "106"},
                {"red", "101"},     {"pink", "105"},     {"yellow", "103"},    {"white", "107"}};
    }

    static string getColoredText(string textcolor) {
        string text_color_code = textColorMap()[textcolor];
        return "\033[" + textColorMap()[textcolor] + "m";
    }
    static string getColoredText(string textColor, string bgColor) {
        return "\033[" + textColorMap()[textColor] + ";" + bgColorMap()[bgColor] + "m";
    }

    static void enableColor(string textColor) {
#if defined(_WIN32)
        static const HANDLE handle = GetStdHandle(STD_OUTPUT_HANDLE);
        int textColorInt = colorToIntMap()[textColor];
        SetConsoleTextAttribute(handle, textColorInt);
#else
        cout << getColoredText(textColor);
#endif
    }
    static void enableColor(string textColor, string bgColor) {
#if defined(_WIN32)
        static const HANDLE handle = GetStdHandle(STD_OUTPUT_HANDLE);
        int textColorInt = colorToIntMap()[textColor];
        int bgColorInt = colorToIntMap()[bgColor];
        int colorAttribute = textColorInt + bgColorInt * 16;
        SetConsoleTextAttribute(handle, colorAttribute);
#else
        cout << getColoredText(textColor, bgColor);
#endif  // Windows/Linux
    }

    static void resetColor() {
#if defined(_WIN32)
        static const HANDLE handle = GetStdHandle(STD_OUTPUT_HANDLE);
        SetConsoleTextAttribute(handle, 7);  // reset color
#else
        cout << "\033[0m";  // reset color
#endif  // Windows/Linux
    }
    static void print(string s, string textColor = "white", string bgColor = "") {
#ifdef _COLORED_LOG
        if (bgColor != "") {
            enableColor(textColor, bgColor);
        } else {
            enableColor(textColor);
        }
        cout << s;
        resetColor();
#endif
    }
    static void println(string s, string textColor = "white", string bgColor = "") {
#ifdef _COLORED_LOG
        print(s, textColor, bgColor);
        cout << endl;
#endif
    }
};

class TimeUtil {
   public:
    static TimePoint now() {
        return std::chrono::steady_clock::now();
    }
    static long long duration(TimePoint start, TimePoint end) {
        return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    }
    static long long elapsed(TimePoint start) {
        return duration(start, now());
    }
};
#endif  // UTIL_H
