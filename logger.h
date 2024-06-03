#pragma once
#include <fstream>
#include <iostream>
#include <numeric>
#include <filesystem>
#include <format>
#include <chrono>

#ifdef _WIN32
#include <windows.h>
#endif

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "global.h"

using namespace std;
using namespace cv;
using namespace std::filesystem;
using namespace std::chrono;

class Logger {
    bool logEnable;
    bool debugMode;
    bool writeLive;  // for internal usage
    int numChannels;
    vector<tm> preTms;
    vector<Mat> canvases;
    ODRecord preOdRcd;

    vector<int> vchStatesPre;  /// vch states (0: non-connected, 1: connected)
    vector<int> maxFireProbs, maxSmokeProbs, accNumFires, accNumSmokes;
    vector<int> maxFireProbsDay, maxSmokeProbsDay, accNumFiresDay, accNumSmokesDay;

    vector<bool> writeChImgs;

    vector<int> preTimes;
    bool pageUpdated[2];
    vector<bool> vchUpdated;

    // for debug
    VideoWriter writer;

   private:
    int toABTime(string &str);
    void newDate(int numChannels, tm *inputTm);
    bool checkDirectories(int numChannels);

    void drawCanvase(Mat &frame, int vchID, tm *curTm, int msec);

    void writeIS(Config &cfg, ODRecord &odRcd, CCRecord &ccRcd);
    void writeCntLine(ofstream &f, CntLine *c, CntLine *p = NULL);
    void writeZone(ofstream &f, Zone *c, Zone *p = NULL);

    void writeNowLive(int sec, int msec, bool boostMode);
    void writeNowOD(ODRecord &odRcd, int vchID, string dayInfo, int sec);
    void writeNowFD(int vchID, string dayInfo, int sec);
    void writeNowCC(CCRecord &ccRcd, int vchID, string dayInfo, int sec);

   public:
    Logger(Config &cfg, ODRecord &odRcd, FDRecord &fdRcd, CCRecord &ccRcd);
    ~Logger();

    static std::ofstream logFile;
    static void writeLog(string msg) {
        // cout << ".";
        cout << msg;
        logFile << msg;
    }

    bool createLog();
    bool needToDraw(int vchID);
    bool checkCmd(Config &cfg, ODRecord &odRcd, FDRecord &fdRcd, CCRecord &ccRcd);
    void writeData(Config &cfg, ODRecord &odRcd, FDRecord &fdRcd, CCRecord &ccRcd, Mat &frame, unsigned int &frameCnt,
                   int vchID, system_clock::time_point now);
};