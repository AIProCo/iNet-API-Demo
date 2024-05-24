#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>

#ifdef _WIN32
#include <concurrent_queue.h>
#else
#include <tbb/concurrent_queue.h>
#endif

class CMat {
   public:
    cv::Mat frame;
    int vchID;
    unsigned int frameCnt;

   public:
    CMat() {
        vchID = -1;
        frameCnt = 0;
    }

    CMat(cv::Mat &frame, int vchID, unsigned int frameCnt) {
        set(frame, vchID, frameCnt);
    }

    void get(cv::Mat &_frame, int &_vchID, unsigned int &_frameCnt) {
        _frame = frame;
        _vchID = vchID;
        _frameCnt = frameCnt;
    }

    // Only one thread/writer can reset/write the counter's value.
    void set(cv::Mat &_frame, int _vchID, unsigned int _frameCnt) {
        frame = _frame;
        vchID = _vchID;
        frameCnt = _frameCnt;
    }
};

typedef concurrency::concurrent_queue<CMat> CMats;
