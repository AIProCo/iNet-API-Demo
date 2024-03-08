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
    int toABTime(string &str) {
        if (str.length() != 8) {
            cout << "Directory Error: " << str << endl;
            return 0;
        }

        int s = stoi(str);
        int month = (s / 100) % 100;
        int abMonth = 0;

        switch (month) {
            case 1:
                abMonth = 0;
                break;
            case 2:
                abMonth = 31;
                break;
            case 3:
                abMonth = 59;
                break;
            case 4:
                abMonth = 90;
                break;
            case 5:
                abMonth = 120;
                break;
            case 6:
                abMonth = 151;
                break;
            case 7:
                abMonth = 181;
                break;
            case 8:
                abMonth = 212;
                break;
            case 9:
                abMonth = 243;
                break;
            case 10:
                abMonth = 273;
                break;
            case 11:
                abMonth = 304;
                break;
            case 12:
                abMonth = 334;
                break;
            default:
                cout << "toABTime error!\n";
                break;
        }
        int abTime = s % 100 + abMonth + (s / 10000) * 365;

        return abTime;
    }

    void newDate(int numChannels, tm *inputTm) {
        tm *curTm;
        if (!inputTm) {
            time_t now = time(NULL);
            curTm = localtime(&now);
        } else
            curTm = inputTm;

        char buf[80];
        strftime(buf, sizeof(buf), "%Y%m%d", curTm);
        string dayInfo = string(buf);

        // writeLog(std::format("New Date: {}", dayInfo));

        vector<string> targetDirs = {CNT_PATH, FD_PATH, CC_PATH};
        int curABTime = toABTime(dayInfo);

        for (const string &targetDir : targetDirs) {
            // create log directories for today
            for (int ch = 0; ch < numChannels; ch++) {
                string dirDate, dirChannel;

                dirDate = targetDir + "/" + dayInfo;
                dirChannel = dirDate + "/" + to_string(ch);

                if (!exists(dirDate))
                    create_directory(dirDate);

                if (!exists(dirChannel))
                    create_directory(dirChannel);

                string dirTxts = dirChannel + "/txts";
                string dirImgs = dirChannel + "/imgs";

                if (!exists(dirTxts))
                    create_directory(dirTxts);

                if (!exists(dirImgs))
                    create_directory(dirImgs);

                if (targetDir == CNT_PATH || targetDir == CC_PATH) {
                    string dirNow = dirChannel + "/now";

                    if (exists(dirNow))  // delete an old now directory
                        remove_all(dirNow);

                    create_directory(dirNow);
                }
            }

            // delete outdated directories
            for (const auto &item : directory_iterator(targetDir)) {
                string dirName = item.path().filename().string();

                if (dirName.length() != 8) {
                    writeLog(std::format("Directory name error. Delete this: {}\n", dirName));
                    remove_all(item);
                    continue;
                }

                int dirABTime = toABTime(dirName);

                if ((curABTime - dirABTime) > 31) {
                    writeLog(std::format("Remove: {}\n", dirName));
                    remove_all(item);
                }
            }
        }

        // delete outdated log files
        for (const auto &item : directory_iterator(LOG_PATH)) {
            string filename = item.path().filename().string();  // with ".txt extension"

            if (filename.length() != 12) {
                writeLog(std::format("Log file name error. Delete this: {}\n", filename));
                remove(item);
                continue;
            }

            filename.resize(8);

            int dirABTime = toABTime(filename);

            if ((curABTime - dirABTime) > 31) {
                writeLog(std::format("Remove: {}.txt\n", filename));
                remove(item);
            }
        }
    }

    bool checkDirectories(int numChannels) {
        vector<string> dirPaths = {AIPRO_PATH, ROOT_PATH, CONFIG_PATH, CHIMGS_PATH,
                                   CNT_PATH,   CC_PATH,   FD_PATH,     LOG_PATH};

        for (string &dirPath : dirPaths) {
            bool isExist = exists(dirPath);

            if (!isExist)
                create_directory(dirPath);
        }

        string chImgsNowPath = string(CHIMGS_PATH) + "/now";

        if (exists(chImgsNowPath))  // delete an old now directory
            remove_all(chImgsNowPath);

        create_directory(chImgsNowPath);

        newDate(numChannels, NULL);

        return true;
    }

    void monitor1x1(Mat &frame, int vchID, tm *curTm, int msec) {
        if (vchUpdated[vchID])
            return;

        Mat &canvase = canvases[0];  // use only the canvase one
        Mat resized;

        resize(frame, canvase, Size(1920, 1080));
        // rectangle(canvase, Rect(0, 0, 42, 28), Scalar(255, 255, 255), -1);
        // putText(canvase, to_string(vchID + 1), Point(0, 24), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 0));

        vchUpdated[vchID] = true;
        pageUpdated[0] = true;
    }

    void monitor2x2(Mat &frame, int vchID, tm *curTm, int msec) {
        if (vchUpdated[vchID])
            return;

        Mat &canvase = canvases[0];  // use only the canvase one
        Mat resized;

        resize(frame, resized, Size(960, 540));
        // rectangle(resized, Rect(0, 0, 42, 28), Scalar(255, 255, 255), -1);
        // putText(resized, to_string(vchID + 1), Point(0, 24), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 0));
        Point tl(vchID % 2 * 960, (vchID / 2) * 540);
        resized.copyTo(canvase(Rect(tl, Size(960, 540))));

        vchUpdated[vchID] = true;
        pageUpdated[0] = true;
    }

    void monitor3x3(Mat &frame, int vchID, tm *curTm, int msec) {
        if (vchUpdated[vchID])
            return;

        int page = (vchID < 9) ? 0 : 1;
        Mat &canvase = canvases[page];
        Mat resized;

        resize(frame, resized, Size(640, 360));
        // rectangle(resized, Rect(0, 0, 42, 28), Scalar(255, 255, 255), -1);
        // putText(resized, to_string(vchID + 1), Point(0, 24), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 0));
        Point tl(vchID % 3 * 640, (vchID / 3) % 3 * 360);
        resized.copyTo(canvase(Rect(tl, Size(640, 360))));

        vchUpdated[vchID] = true;
        pageUpdated[page] = true;
    }

    void writeIS(Config &cfg, ODRecord &odRcd, CCRecord &ccRcd) {
        string filenameIS = "is.txt";
        string txtPathIS = string(CONFIG_PATH) + "/" + filenameIS;

        ofstream is(txtPathIS);
        writeLog(std::format("writeIS: Line = {}, Zone = {}, CZone = {}\n", odRcd.cntLines.size(), odRcd.zones.size(),
                             ccRcd.ccZones.size()));

        if (is.is_open()) {
            for (auto &cntLine : odRcd.cntLines) {
                is << 0 << " ";
                is << cntLine.clineID << " ";
                is << cntLine.vchID << " ";
                is << cntLine.isMode << " ";
                is << cntLine.pts[0].x << " ";
                is << cntLine.pts[0].y << " ";
                is << cntLine.pts[1].x << " ";
                is << cntLine.pts[1].y << endl;
            }
            for (auto &zone : odRcd.zones) {
                is << 1 << " ";
                is << zone.zoneID << " ";
                is << zone.vchID << " ";
                is << zone.isMode << " ";
                is << zone.pts[0].x << " ";
                is << zone.pts[0].y << " ";
                is << zone.pts[1].x << " ";
                is << zone.pts[1].y << " ";
                is << zone.pts[2].x << " ";
                is << zone.pts[2].y << " ";
                is << zone.pts[3].x << " ";
                is << zone.pts[3].y << endl;
            }
            for (auto &ccZone : ccRcd.ccZones) {
                is << 2 << " ";
#ifndef _CPU_INFER
                is << ccZone.ccZoneID << " ";
                is << ccZone.vchID << " ";
                is << ccZone.pts[0].x << " ";
                is << ccZone.pts[0].y << " ";
                is << ccZone.pts[1].x << " ";
                is << ccZone.pts[1].y << " ";
                is << ccZone.pts[2].x << " ";
                is << ccZone.pts[2].y << " ";
                is << ccZone.pts[3].x << " ";
                is << ccZone.pts[3].y << " ";
#else
                is << ccZone.vchID << " ";
#endif
                is << ccZone.ccLevelThs[0] << " ";
                is << ccZone.ccLevelThs[1] << " ";
                is << ccZone.ccLevelThs[2] << endl;
            }

            // Write scoreThs
            is << 3 << " ";
            is << int(cfg.odScoreTh * 1000) << " ";
            is << int(cfg.fdScoreTh * 1000) << endl;
        }

        is.close();
    }

    void writeCntLine(ofstream &f, CntLine *c, CntLine *p = NULL) {
        /*if (p)
            if (p->totalUL[0][0] == 0 && p->totalUL[0][1] == 0 && p->totalUL[0][2] == 0 && p->totalUL[1][0] == 0 &&
                p->totalUL[1][1] == 0 && p->totalUL[1][2] == 0 && p->totalDR[0][0] == 0 && p->totalDR[0][1] == 0 &&
                p->totalDR[0][2] == 0 && p->totalDR[1][0] == 0 && p->totalDR[1][1] == 0 && p->totalDR[1][2] == 0)
                f << "inited\n";*/

        f << 0 << " " << c->clineID << " " << c->isMode << " ";
        for (int i = 0; i < NUM_GENDERS; i++)
            for (int j = 0; j < NUM_AGE_GROUPS; j++)
                if (p)
                    f << (c->totalUL[i][j] - p->totalUL[i][j]) << " ";
                else
                    f << c->totalUL[i][j] << " ";

        for (int i = 0; i < NUM_GENDERS; i++)
            for (int j = 0; j < NUM_AGE_GROUPS; j++)
                if (p)
                    f << (c->totalDR[i][j] - p->totalDR[i][j]) << " ";
                else
                    f << c->totalDR[i][j] << " ";

        f << endl;
    }

    void writeZone(ofstream &f, Zone *c, Zone *p = NULL) {
        /*if (p)
            if (p->hitMap[0][0] == 0 && p->hitMap[0][1] == 0 && p->hitMap[0][2] == 0 && p->hitMap[1][0] == 0 &&
                p->hitMap[1][1] == 0 && p->hitMap[1][2] == 0)
                f << "inited\n";*/

        f << 1 << " " << c->zoneID << " " << c->isMode << " ";
        for (int i = 0; i < NUM_GENDERS; i++)
            for (int j = 0; j < NUM_AGE_GROUPS; j++)
                f << c->curPeople[i][j] << " ";

        for (int i = 0; i < NUM_GENDERS; i++)
            for (int j = 0; j < NUM_AGE_GROUPS; j++)
                if (p)
                    f << (c->hitMap[i][j] - p->hitMap[i][j]) << " ";
                else
                    f << c->hitMap[i][j] << " ";

        f << endl;
    }

   public:
    static std::ofstream logFile;

    Logger(Config &cfg, ODRecord &odRcd, FDRecord &fdRcd, CCRecord &ccRcd) {
        logEnable = cfg.logEnable;
        numChannels = cfg.numChannels;
        preOdRcd = odRcd;
        debugMode = cfg.debugMode;

        // maximum two pages(18 channels)
        pageUpdated[0] = false, pageUpdated[1] = false;
        preTimes.resize(2, -1);

        if (!logEnable)
            return;

        vchStatesPre.resize(numChannels, 0);

        maxFireProbs.resize(numChannels, 0);
        maxSmokeProbs.resize(numChannels, 0);
        accNumFires.resize(numChannels, 0);
        accNumSmokes.resize(numChannels, 0);

        maxFireProbsDay.resize(numChannels, 0);
        maxSmokeProbsDay.resize(numChannels, 0);
        accNumFiresDay.resize(numChannels, 0);
        accNumSmokesDay.resize(numChannels, 0);

        writeChImgs.resize(numChannels, true);
        vchUpdated.resize(numChannels, false);

        for (int i = 0; i < numChannels; i++) {
            preTms.push_back(tm(-1, -1, -1, -1, -1, -1, -1, -1, -1));
            canvases.push_back(Mat::zeros(1080, 1920, CV_8UC3));
        }

        if (!checkDirectories(numChannels)) {
            cout << "Parsing Error!\n";
        }

        // checkCmd(cfg, odRcd, fdRcd, ccRcd);
        // writeIS(cfg, odRcd, ccRcd);

        createLog();
        cfg.lg = Logger::writeLog;  // for writing log in other classes

        // for debug
        // if (debugMode)
        //    writer.open("out.mp4", VideoWriter::fourcc('m', 'p', '4', 'v'), 5, Size(1920, 1080));
    }

    ~Logger() {
        if (logFile.is_open())
            logFile.close();

        // if (debugMode)
        //    writer.release();
    }

    bool createLog() {
        if (logFile.is_open())
            logFile.close();

        // create a log file
        system_clock::time_point now = system_clock::now();
        time_t nowt = system_clock::to_time_t(now);
        tm *curTm = localtime(&nowt);

        char buf[80];
        strftime(buf, sizeof(buf), "%Y%m%d", curTm);
        string dayInfo = string(buf);

        strftime(buf, sizeof(buf), "%T", curTm);
        string timeInfo = string(buf);

        // create a log file
        string logFilepath = string(LOG_PATH) + "/" + dayInfo + ".txt";
        logFile.open(logFilepath, ios_base::app);

        if (!logFile.is_open()) {
            cout << "Log file error:" << logFilepath << endl;
            return false;
        }

        writeLog(std::format("\nStart logging {} at {} ------------------------\n", logFilepath, timeInfo));
        return true;
    }

    static void writeLog(string msg) {
        // cout << ".";
        cout << msg;
        logFile << msg;
    }

    bool needToDraw(int vchID) {
        return !vchUpdated[vchID];
    }

    bool checkCmd(Config &cfg, ODRecord &odRcd, FDRecord &fdRcd, CCRecord &ccRcd) {
        if (!logEnable)
            return true;

        string filenameS2E = "cmds2e.txt";
        string txtPathS2E = string(CONFIG_PATH) + "/" + filenameS2E;
        ifstream cmdFileS2E(txtPathS2E);
        bool updateIS = false;
        bool terminateProgram = false;

        if (cmdFileS2E.is_open()) {
            updateIS = true;
            string line;

            while (getline(cmdFileS2E, line)) {
                int cmd;
                stringstream ss(line);

                ss >> cmd;
                writeLog(std::format("Get cmd: {}\n", line));

                switch (cmd) {
                    case CMD_INSERT_LINE: {
                        CntLine cntLine;

                        cntLine.enabled = true;
                        ss >> cntLine.clineID;
                        ss >> cntLine.vchID;
                        ss >> cntLine.isMode;
                        ss >> cntLine.pts[0].x;
                        ss >> cntLine.pts[0].y;
                        ss >> cntLine.pts[1].x;
                        ss >> cntLine.pts[1].y;

                        int fH = cfg.frameHeights[cntLine.vchID];
                        int fW = cfg.frameWidths[cntLine.vchID];

                        if (cntLine.pts[0].x < 0 || cntLine.pts[0].x >= fW || cntLine.pts[1].x < 0 ||
                            cntLine.pts[1].x >= fW) {
                            writeLog(
                                std::format("cntLine.pts.x error: {} {} {}", cntLine.pts[0].x, cntLine.pts[1].x, fW));
                            return false;
                        }

                        if (cntLine.pts[0].y < 0 || cntLine.pts[0].y >= fH || cntLine.pts[1].y < 0 ||
                            cntLine.pts[1].y >= fH) {
                            writeLog(
                                std::format("cntLine.pts.y error: {} {} {}", cntLine.pts[0].y, cntLine.pts[1].y, fH));
                            return false;
                        }

                        if (abs(cntLine.pts[0].x - cntLine.pts[1].x) > abs(cntLine.pts[0].y - cntLine.pts[1].y))
                            cntLine.direction = 0;  // horizontal line -> use delta y and count U and D
                        else
                            cntLine.direction = 1;  // vertical line -> use delta x and count L and R

                        for (int g = 0; g < NUM_GENDERS; g++)
                            for (int a = 0; a < NUM_AGE_GROUPS; a++)
                                cntLine.totalUL[g][a] = cntLine.totalDR[g][a] = 0;

                        bool duplicated = false;
                        for (auto &item : odRcd.cntLines)
                            if (item.vchID == cntLine.vchID && item.clineID == cntLine.clineID) {
                                duplicated = true;
                                break;
                            }

                        if (!duplicated) {
                            odRcd.cntLines.push_back(cntLine);
                            preOdRcd.cntLines.push_back(cntLine);
                        }
                        break;
                    }
                    case CMD_REMOVE_LINE: {
                        int cLineID, vchID;
                        ss >> cLineID;
                        ss >> vchID;

                        if (vchID >= cfg.numChannels) {
                            writeLog(std::format("vchID error in remove line: {} {}", vchID, cfg.numChannels));
                            return false;
                        }

                        for (auto itr = odRcd.cntLines.begin(); itr != odRcd.cntLines.end();) {
                            if (itr->clineID == cLineID && itr->vchID == vchID)
                                itr = odRcd.cntLines.erase(itr);
                            else
                                ++itr;
                        }

                        for (auto itr = preOdRcd.cntLines.begin(); itr != preOdRcd.cntLines.end();) {
                            if (itr->clineID == cLineID && itr->vchID == vchID)
                                itr = preOdRcd.cntLines.erase(itr);
                            else
                                ++itr;
                        }
                        break;
                    }
                    case CMD_INSERT_ZONE: {
                        Zone zone;

                        zone.enabled = true;
                        ss >> zone.zoneID;
                        ss >> zone.vchID;
                        ss >> zone.isMode;

                        if (zone.vchID >= cfg.numChannels) {
                            writeLog(std::format("zone.vchID error: {} {}", zone.vchID, cfg.numChannels));
                            return false;
                        }

                        int fH = cfg.frameHeights[zone.vchID];
                        int fW = cfg.frameWidths[zone.vchID];

                        for (int i = 0; i < 4; i++) {
                            Point pt;
                            ss >> pt.x;
                            ss >> pt.y;
                            zone.pts.push_back(pt);

                            if (pt.x < 0 || pt.x >= fW) {
                                writeLog(std::format("zone.pts.x error: {} {}", pt.x, fW));
                                return false;
                            }

                            if (pt.y < 0 || pt.y >= fH) {
                                writeLog(std::format("zone.pts.y error: {} {}", pt.y, fH));
                                return false;
                            }
                        }

                        for (int g = 0; g < NUM_GENDERS; g++)
                            for (int a = 0; a < NUM_AGE_GROUPS; a++)
                                zone.hitMap[g][a] = 0;

                        bool duplicated = false;
                        for (auto &item : odRcd.zones)
                            if (item.vchID == zone.vchID && item.zoneID == zone.zoneID) {
                                duplicated = true;
                                break;
                            }

                        if (!duplicated) {
                            odRcd.zones.push_back(zone);
                            preOdRcd.zones.push_back(zone);
                        }

                        break;
                    }
                    case CMD_REMOVE_ZONE: {
                        int zoneID, vchID;
                        ss >> zoneID;
                        ss >> vchID;

                        if (vchID >= cfg.numChannels) {
                            writeLog(std::format("vchID error in remove zone: {} {}", vchID, cfg.numChannels));
                            return false;
                        }

                        for (auto itr = odRcd.zones.begin(); itr != odRcd.zones.end();) {
                            if (itr->zoneID == zoneID && itr->vchID == vchID)
                                itr = odRcd.zones.erase(itr);
                            else
                                ++itr;
                        }

                        for (auto itr = preOdRcd.zones.begin(); itr != preOdRcd.zones.end();) {
                            if (itr->zoneID == zoneID && itr->vchID == vchID)
                                itr = preOdRcd.zones.erase(itr);
                            else
                                ++itr;
                        }
                        break;
                    }
                    case CMD_INSERT_CCZONE: {
                        CCZone ccZone;

                        ccZone.enabled = true;
                        ccZone.ccLevel = 0;
#ifndef _CPU_INFER
                        ss >> ccZone.ccZoneID;
                        ss >> ccZone.vchID;

                        if (ccZone.vchID >= cfg.numChannels) {
                            writeLog(std::format("ccZone.vchID error: {} {}", ccZone.vchID, cfg.numChannels));
                            return false;
                        }

                        int fH = cfg.frameHeights[ccZone.vchID];
                        int fW = cfg.frameWidths[ccZone.vchID];

                        for (int i = 0; i < 4; i++) {
                            Point pt;
                            ss >> pt.x;
                            ss >> pt.y;
                            ccZone.pts.push_back(pt);

                            if (pt.x < 0 || pt.x >= fW) {
                                writeLog(std::format("zone.pts.x error: {} {}", pt.x, fW));
                                return false;
                            }

                            if (pt.y < 0 || pt.y >= fH) {
                                writeLog(std::format("zone.pts.y error: {} {}", pt.y, fH));
                                return false;
                            }
                        }
#else
                        ccZone.ccZoneID = -1;
                        ss >> ccZone.vchID;

                        if (ccZone.vchID >= cfg.numChannels) {
                            writeLog(std::format("ccZone.vchID error: {} {}", ccZone.vchID, cfg.numChannels));
                            return false;
                        }

#endif
                        ss >> ccZone.ccLevelThs[0];
                        ss >> ccZone.ccLevelThs[1];
                        ss >> ccZone.ccLevelThs[2];

                        ccZone.ccNums.resize(cfg.ccWindowSize, 0);

                        ccZone.maxCC = 0;
                        ccZone.maxCCDay = 0;

                        for (int i = 0; i < NUM_CC_LEVELS - 1; i++) {
                            ccZone.accCCLevels[i] = 0;
                            ccZone.accCCLevelsDay[i] = 0;
                        }

                        // init mask with empty Mat. This is generated in CrowdCounter::runModel.
                        ccZone.mask = cv::Mat();

                        bool duplicated = false;
                        for (auto &item : ccRcd.ccZones)
                            if (item.vchID == ccZone.vchID && item.ccZoneID == ccZone.ccZoneID) {
                                duplicated = true;
                                break;
                            }

                        if (!duplicated)
                            ccRcd.ccZones.push_back(ccZone);

                        break;
                    }
                    case CMD_REMOVE_CCZONE: {
                        int ccZoneID, vchID;
#ifndef _CPU_INFER
                        ss >> ccZoneID;
#else
                        ccZoneID = -1;
#endif
                        ss >> vchID;

                        if (vchID >= cfg.numChannels) {
                            writeLog(std::format("vchID error in remove cczone: {} {}", vchID, cfg.numChannels));
                            return false;
                        }

                        for (auto itr = ccRcd.ccZones.begin(); itr != ccRcd.ccZones.end();) {
                            if (itr->ccZoneID == ccZoneID && itr->vchID == vchID)
                                itr = ccRcd.ccZones.erase(itr);
                            else
                                ++itr;
                        }
                        break;
                    }
                    case CMD_UPDATE_SCORETHS: {
                        int odScoreTh, fdScoreTh;
                        ss >> odScoreTh;
                        ss >> fdScoreTh;

                        if (odScoreTh >= 1000 || fdScoreTh >= 1000) {
                            writeLog(std::format("ScoreThs error: {} {}", odScoreTh, fdScoreTh));
                            return false;
                        }

                        cfg.odScoreTh = (odScoreTh / 1000.0f);
                        cfg.fdScoreTh = (fdScoreTh / 1000.0f);

                        break;
                    }
                    case CMD_CLEARLOG: {
                        remove_all(CNT_PATH);
                        remove_all(FD_PATH);
                        remove_all(CC_PATH);

                        checkDirectories(numChannels);

                        for (auto &zone : odRcd.zones)
                            for (int g = 0; g < NUM_GENDERS; g++)
                                for (int a = 0; a < NUM_AGE_GROUPS; a++)
                                    zone.hitMap[g][a] = 0;

                        for (auto &cntLine : odRcd.cntLines)
                            for (int g = 0; g < NUM_GENDERS; g++)
                                for (int a = 0; a < NUM_AGE_GROUPS; a++)
                                    cntLine.totalUL[g][a] = cntLine.totalDR[g][a] = 0;

                        for (int i = 0; i < cfg.numChannels; i++) {
                            fdRcd.fireProbsMul[i].clear();
                            fdRcd.fireProbsMul[i].clear();

                            fdRcd.fireProbsMul[i].resize(cfg.fdWindowSize, 0.0f);
                            fdRcd.smokeProbsMul[i].resize(cfg.fdWindowSize, 0.0f);
                        }

                        for (int i = 0; i < cfg.numChannels; i++) {
                            ccRcd.ccNumFrames[i].clear();
                            ccRcd.ccNumFrames[i].resize(cfg.ccWindowSize);
                        }

                        for (CCZone &ccZone : ccRcd.ccZones) {
                            ccZone.ccNums.clear();
                            ccZone.ccNums.resize(cfg.ccWindowSize, 0);

                            ccZone.maxCC = 0;
                            ccZone.maxCCDay = 0;

                            for (int i = 0; i < NUM_CC_LEVELS - 1; i++) {
                                ccZone.accCCLevels[i] = 0;
                                ccZone.accCCLevelsDay[i] = 0;
                            }
                        }

                        break;
                    }
                    case CMD_REMOVE_ALL_LINES_ZONES:
                        odRcd.cntLines.clear();
                        odRcd.zones.clear();
                        break;
                    case CMD_REMOVE_ALL_LINES:
                        odRcd.cntLines.clear();
                        break;
                    case CMD_REMOVE_ALL_ZONES:
                        odRcd.zones.clear();
                        break;
                    case CMD_REMOVE_ALL_CCZONES:
                        ccRcd.ccZones.clear();
                        break;
                    case CMD_TERMINATE_PROGRAM:
                        terminateProgram = true;
                        break;
                    default:
                        break;
                }
            }
        }

        cmdFileS2E.close();
        remove(txtPathS2E);

        if (terminateProgram) {
            return false;
        } else if (updateIS) {
            writeIS(cfg, odRcd, ccRcd);
            writeChImgs.clear();
            writeChImgs.resize(cfg.numChannels, true);
        }

        return true;
    }

    void writeData(Config &cfg, ODRecord &odRcd, FDRecord &fdRcd, CCRecord &ccRcd, Mat &frame, unsigned int &frameCnt,
                   vector<FireBox> &fboxes, int vchID, system_clock::time_point now) {
        if (!logEnable)
            return;

        // write cmde2s.txt to update vch states
        if (vchStatesPre != cfg.vchStates) {
            string filename = "cmde2s.txt";
            string txtPathCmde2s = string(CONFIG_PATH) + "/" + filename;
            ofstream logFileCmde2s(txtPathCmde2s);

            if (logFileCmde2s.is_open()) {
                vchStatesPre = cfg.vchStates;
                logFileCmde2s << "0 " << numChannels;
                writeLog("0 ");

                for (int &state : vchStatesPre) {
                    logFileCmde2s << " " << state;
                    writeLog(std::format(" {}", state));
                }

                writeLog(" <= Write cmde2s.txt\n");
                logFileCmde2s.close();
            }
        }

        time_t nowt = system_clock::to_time_t(now);
        tm *curTm = localtime(&nowt);
        tm *preTm = &preTms[vchID];

        auto duration = now.time_since_epoch();
        auto millisObj = duration_cast<milliseconds>(duration) % 1000;
        int msec = millisObj.count();

        char buf[80];
        strftime(buf, sizeof(buf), "%Y%m%d", curTm);
        string dayInfo = string(buf);
        string dayInfoPre;
        bool dayChanged = false;

        /// for chimgs
        if (writeChImgs[vchID]) {
            string filename = std::format("{}.jpg", vchID);
            string imgPath = string(CHIMGS_PATH) + "/" + filename;
            imwrite(imgPath, frame);

            writeChImgs[vchID] = false;
        }

        if (numChannels > 4)
            monitor3x3(frame, vchID, curTm, msec);
        else if (numChannels > 1)
            monitor2x2(frame, vchID, curTm, msec);
        else
            monitor1x1(frame, vchID, curTm, msec);

        // generate a 30 min log and a one day log
        // if ((preTm.tm_min == 29 && curTm->tm_min == 30) || (preTm.tm_min != -1 && preTm.tm_min != curTm->tm_min)){
        if ((preTm->tm_min == 29 && curTm->tm_min == 30) || (preTm->tm_min == 59 && curTm->tm_min == 0)) {
            int &accNumFire = accNumFires[vchID];
            int &accNumSmoke = accNumSmokes[vchID];
            int &maxFireProb = maxFireProbs[vchID];
            int &maxSmokeProb = maxSmokeProbs[vchID];

            int &accNumFireDay = accNumFiresDay[vchID];
            int &accNumSmokeDay = accNumSmokesDay[vchID];
            int &maxFireProbDay = maxFireProbsDay[vchID];
            int &maxSmokeProbDay = maxSmokeProbsDay[vchID];

            string filename, txtPathFD, txtPathCnt, txtPathCC, dayInfoSelected;

            /// for cnt
            if (preTm->tm_mday == curTm->tm_mday) {
                filename = std::format("{:02}{:02}.txt", curTm->tm_hour, curTm->tm_min);
                dayInfoSelected = dayInfo;
            } else {
                filename = "2400.txt";

                strftime(buf, sizeof(buf), "%Y%m%d", preTm);
                dayInfoPre = string(buf);
                dayInfoSelected = dayInfoPre;
            }

            txtPathCnt = string(CNT_PATH) + "/" + dayInfoSelected + "/" + to_string(vchID) + "/" + filename;
            txtPathFD = string(FD_PATH) + "/" + dayInfoSelected + "/" + to_string(vchID) + "/" + filename;
            txtPathCC = string(CC_PATH) + "/" + dayInfoSelected + "/" + to_string(vchID) + "/" + filename;

            if (cfg.odChannels[vchID]) {
                ofstream logFileCnt(txtPathCnt);

                if (logFileCnt.is_open()) {
                    for (int n = 0; n < odRcd.cntLines.size(); n++) {
                        if (odRcd.cntLines[n].vchID != vchID)
                            continue;

                        auto &c = odRcd.cntLines[n];
                        auto &p = preOdRcd.cntLines[n];

                        writeCntLine(logFileCnt, &c, &p);
                        p = c;
                    }

                    for (int n = 0; n < odRcd.zones.size(); n++) {
                        if (odRcd.zones[n].vchID != vchID)
                            continue;

                        auto &c = odRcd.zones[n];
                        auto &p = preOdRcd.zones[n];

                        writeZone(logFileCnt, &c, &p);
                        p = c;
                    }
                }

                logFileCnt.close();
            }

            if (cfg.fdChannels[vchID]) {
                ofstream logFileFD(txtPathFD);

                if (logFileFD.is_open()) {
                    // cout << vchID << " " << accNumFire << " " << accNumSmoke << " " << maxFireProb << " " <<
                    // maxSmokeProb<<endl;
                    logFileFD << accNumFire << " " << accNumSmoke << " " << maxFireProb << " " << maxSmokeProb;
                }

                logFileFD.close();

                // update parameters for day fd file
                if (maxFireProb > maxFireProbDay)
                    maxFireProbDay = maxFireProb;

                if (maxSmokeProb > maxSmokeProbDay)
                    maxSmokeProbDay = maxSmokeProb;

                accNumFireDay += accNumFire;
                accNumSmokeDay += accNumSmoke;

                accNumFire = accNumSmoke = maxFireProb = maxSmokeProb = 0;
            }

            if (cfg.ccChannels[vchID]) {
                ofstream logFileCC(txtPathCC);

                if (logFileCC.is_open()) {
                    for (CCZone &ccZone : ccRcd.ccZones) {
                        if (ccZone.vchID != vchID)
                            continue;

                        string text = std::format("{} {} {} {} {}\n", ccZone.ccZoneID, ccZone.maxCC,
                                                  ccZone.accCCLevels[0], ccZone.accCCLevels[1], ccZone.accCCLevels[2]);
                        logFileCC << text;

                        // update parameters for day cc file
                        if (ccZone.maxCC > ccZone.maxCCDay)
                            ccZone.maxCCDay = ccZone.maxCC;

                        ccZone.maxCC = 0;

                        for (int i = 0; i < NUM_CC_LEVELS - 1; i++) {
                            ccZone.accCCLevelsDay[i] += ccZone.accCCLevels[i];
                            ccZone.accCCLevels[i] = 0;
                        }
                    }

                    logFileCC.close();
                }
            }

            // one day log
            // if (curTm->tm_min % 2 == 0) { dayInfoPre = dayInfo;
            if (preTm->tm_mday != curTm->tm_mday) {
                if (vchID == 0)  // just once for all vchIDs
                    dayChanged = true;

                newDate(cfg.numChannels, curTm);  // prepare a new directory for curTm

                string filename = "day.txt";
                string txtPathCnt = string(CNT_PATH) + "/" + dayInfoPre + "/" + to_string(vchID) + "/" + filename;
                string txtPathFD = string(FD_PATH) + "/" + dayInfoPre + "/" + to_string(vchID) + "/" + filename;
                string txtPathCC = string(CC_PATH) + "/" + dayInfoPre + "/" + to_string(vchID) + "/" + filename;

                if (cfg.odChannels[vchID]) {
                    ofstream logFileCnt(txtPathCnt);

                    if (logFileCnt.is_open()) {
                        for (int n = 0; n < odRcd.cntLines.size(); n++) {
                            if (odRcd.cntLines[n].vchID != vchID)
                                continue;

                            auto &c = odRcd.cntLines[n];
                            auto &p = preOdRcd.cntLines[n];

                            writeCntLine(logFileCnt, &c);

                            for (int i = 0; i < 2; i++)
                                for (int j = 0; j < 3; j++) {
                                    c.totalUL[i][j] = c.totalDR[i][j] = 0;
                                    p.totalUL[i][j] = p.totalDR[i][j] = 0;
                                }
                        }

                        for (int n = 0; n < odRcd.zones.size(); n++) {
                            if (odRcd.zones[n].vchID != vchID)
                                continue;

                            auto &c = odRcd.zones[n];
                            auto &p = preOdRcd.zones[n];

                            writeZone(logFileCnt, &c);

                            for (int i = 0; i < 2; i++)
                                for (int j = 0; j < 3; j++)
                                    c.hitMap[i][j] = p.hitMap[i][j] = 0;
                        }
                    }

                    logFileCnt.close();
                }

                if (cfg.fdChannels[vchID]) {
                    ofstream logFileFD(txtPathFD);

                    if (logFileFD.is_open())
                        logFileFD << accNumFireDay << " " << accNumSmokeDay << " " << maxFireProbDay << " "
                                  << maxSmokeProbDay;

                    logFileFD.close();

                    accNumFireDay = accNumSmokeDay = maxFireProbDay = maxSmokeProbDay = 0;
                }

                if (cfg.ccChannels[vchID]) {
                    ofstream logFileCC(txtPathCC);

                    if (logFileCC.is_open()) {
                        for (CCZone &ccZone : ccRcd.ccZones) {
                            if (ccZone.vchID != vchID)
                                continue;

                            string text = std::format("{} {} {} {} {}\n", ccZone.ccZoneID, ccZone.maxCCDay,
                                                      ccZone.accCCLevelsDay[0], ccZone.accCCLevelsDay[1],
                                                      ccZone.accCCLevelsDay[2]);

                            logFileCC << text;

                            ccZone.maxCCDay = 0;
                            for (int i = 0; i < NUM_CC_LEVELS - 1; i++)
                                ccZone.accCCLevelsDay[i] = 0;
                        }

                        logFileCC.close();
                    }
                }
            }
        }

        if (preTm->tm_sec != curTm->tm_sec) {
            if (cfg.odChannels[vchID]) {
                string filenameCnt = std::format("{}.txt", curTm->tm_sec % 10);
                string txtPathCnt = string(CNT_PATH) + "/" + dayInfo + "/" + to_string(vchID) + "/now/" + filenameCnt;
                ofstream logFileCnt(txtPathCnt);

                string filenameRet = std::format("{:02}{:02}{:02}.txt", curTm->tm_hour, curTm->tm_min, curTm->tm_sec);
                string txtPathRet = string(CNT_PATH) + "/" + dayInfo + "/" + to_string(vchID) + "/txts/" + filenameRet;

                if (logFileCnt.is_open()) {
                    bool saveImg = false;

                    for (int n = 0; n < odRcd.cntLines.size(); n++) {
                        if (odRcd.cntLines[n].vchID != vchID)
                            continue;

                        auto &c = odRcd.cntLines[n];
                        writeCntLine(logFileCnt, &c);

                        if (c.isMode == IS_RESTRICTED_AREA) {
                            int curTotal = 0;
                            for (int g = 0; g < NUM_GENDERS; g++)
                                for (int a = 0; a < NUM_AGE_GROUPS; a++)
                                    curTotal += (c.totalUL[g][a] + c.totalDR[g][a]);

                            if (c.preTotal != curTotal) {
                                ofstream logFileRet(txtPathRet);

                                if (logFileRet.is_open()) {
                                    writeCntLine(logFileRet, &c);
                                    logFileRet.close();

                                    saveImg = true;
                                }

                                c.preTotal = curTotal;
                            }
                        }
                    }

                    for (int n = 0; n < odRcd.zones.size(); n++) {
                        if (odRcd.zones[n].vchID != vchID)
                            continue;

                        auto &z = odRcd.zones[n];
                        writeZone(logFileCnt, &z);

                        if (z.isMode == IS_RESTRICTED_AREA) {
                            int curTotal = 0;
                            for (int g = 0; g < NUM_GENDERS; g++)
                                for (int a = 0; a < NUM_AGE_GROUPS; a++)
                                    curTotal += z.curPeople[g][a];

                            if (z.preTotal == 0 && curTotal > 0) {
                                ofstream logFileRet(txtPathRet, ios_base::app);

                                if (logFileRet.is_open()) {
                                    writeZone(logFileRet, &z);
                                    logFileRet.close();

                                    saveImg = true;
                                }
                            }

                            z.preTotal = curTotal;
                        }
                    }

                    if (saveImg) {
                        string filenameRetImg =
                            std::format("{:02}{:02}{:02}.jpg", curTm->tm_hour, curTm->tm_min, curTm->tm_sec);
                        string imgPathRet =
                            string(CNT_PATH) + "/" + dayInfo + "/" + to_string(vchID) + "/imgs/" + filenameRetImg;

                        imwrite(imgPathRet, frame, {cv::ImwriteFlags::IMWRITE_JPEG_QUALITY, 80});
                    }

                    logFileCnt.close();
                }
            }

            if (cfg.fdChannels[vchID]) {
                int numFire = 0, numSmoke = 0;

                for (auto &fbox : fboxes) {
                    if (fbox.prob < cfg.fdScoreTh)
                        continue;  // should check scores are ordered. Otherwise, use continue

                    if (fbox.objID == FIRE)  // draw only the fire
                        numFire++;

                    if (fbox.objID == SMOKE)  // draw only the smoke
                        numSmoke++;
                }

                int fireProb = fdRcd.fireProbsMul[vchID].back() * 1000;
                int smokeProb = fdRcd.smokeProbsMul[vchID].back() * 1000;
                int &afterFireEvent = fdRcd.afterFireEvents[vchID];

                if (numFire || numSmoke || fireProb || smokeProb) {
                    string filename = std::format("{:02}{:02}{:02}.txt", curTm->tm_hour, curTm->tm_min, curTm->tm_sec);
                    string txtPathFD = string(FD_PATH) + "/" + dayInfo + "/" + to_string(vchID) + "/txts/" + filename;
                    ofstream logFileFD(txtPathFD);
                    int isFirst = (afterFireEvent == 0) ? 1 : 0;

                    if (logFileFD.is_open()) {
                        logFileFD << isFirst << " " << numFire << " " << numSmoke << " " << fireProb << " "
                                  << smokeProb;
                        logFileFD.close();
                    }

                    if (isFirst || curTm->tm_sec % 10 == 0) {
                        // if (curTm->tm_sec % 10 == 0) {
                        string filenameImg =
                            std::format("{:02}{:02}{:02}.jpg", curTm->tm_hour, curTm->tm_min, curTm->tm_sec);
                        string imgPathFD =
                            string(FD_PATH) + "/" + dayInfo + "/" + to_string(vchID) + "/imgs/" + filenameImg;

                        imwrite(imgPathFD, frame, {cv::ImwriteFlags::IMWRITE_JPEG_QUALITY, 80});
                    }

                    ++afterFireEvent;
                } else {
                    afterFireEvent = 0;
                }

                // update parameters for 30 min file
                if (fireProb > maxFireProbs[vchID])
                    maxFireProbs[vchID] = fireProb;

                if (smokeProb > maxSmokeProbs[vchID])
                    maxSmokeProbs[vchID] = smokeProb;

                accNumFires[vchID] += numFire;
                accNumSmokes[vchID] += numSmoke;
            }

            if (cfg.ccChannels[vchID]) {
                string filenameCCNow = std::format("{}.txt", curTm->tm_sec % 10);
                string txtPathCCNow =
                    string(CC_PATH) + "/" + dayInfo + "/" + to_string(vchID) + "/now/" + filenameCCNow;
                ofstream logFileCCNow(txtPathCCNow);

                vector<string> ccLogs;
                for (CCZone &ccZone : ccRcd.ccZones) {
                    if (ccZone.vchID == vchID) {
                        int curCCNum = ccZone.ccNums.back();

                        // update parameters for 30 min file
                        if (curCCNum > ccZone.maxCC)
                            ccZone.maxCC = curCCNum;

                        if (logFileCCNow.is_open()) {
                            string text = std::format("{} {} {} {}\n", ccZone.ccZoneID, curCCNum, ccZone.ccLevel,
                                                      ccZone.preCCLevel);
                            logFileCCNow << text;
                        }

                        if (ccZone.ccLevel > 0) {
                            if (ccZone.ccLevel > ccZone.preCCLevel) {
                                string text = std::format("{} {} {} {}\n", ccZone.ccZoneID, curCCNum, ccZone.ccLevel,
                                                          ccZone.preCCLevel);
                                ccLogs.push_back(text);
                            }

                            ccZone.accCCLevels[ccZone.ccLevel - 1]++;
                        }

                        ccZone.preCCLevel = ccZone.ccLevel;
                    }
                }

                if (logFileCCNow.is_open())
                    logFileCCNow.close();

                if (ccLogs.size() > 0) {
                    string filename = std::format("{:02}{:02}{:02}.txt", curTm->tm_hour, curTm->tm_min, curTm->tm_sec);
                    string txtPathCC = string(CC_PATH) + "/" + dayInfo + "/" + to_string(vchID) + "/txts/" + filename;
                    ofstream logFileCC(txtPathCC);

                    if (logFileCC.is_open()) {
                        for (string &ccLog : ccLogs)
                            logFileCC << ccLog;

                        logFileCC.close();
                    }

                    string filenameImg =
                        std::format("{:02}{:02}{:02}.jpg", curTm->tm_hour, curTm->tm_min, curTm->tm_sec);
                    string imgPathCC =
                        string(CC_PATH) + "/" + dayInfo + "/" + to_string(vchID) + "/imgs/" + filenameImg;
                    imwrite(imgPathCC, frame, {cv::ImwriteFlags::IMWRITE_JPEG_QUALITY, 80});
                }
            }
        }

        *preTm = *curTm;

        for (int p = 0; p < 2; p++) {
            if (pageUpdated[p]) {
                int msec5 = msec / 125;  // save at 5 fps

                if (preTimes[p] != msec5) {
                    preTimes[p] = msec5;

                    // string filename = std::format("{}p{:02}{}.jpg", p, curTm->tm_sec, msec5);
                    string filename = std::format("{}p{}{}.jpg", p, curTm->tm_sec % 10, msec5);
                    string imgPath = string(CHIMGS_PATH) + "/now/" + filename;

                    // for debug
                    if (debugMode) {
                        rectangle(canvases[p], Rect(0, 0, 200, 28), Scalar(255, 255, 255), -1);
                        putText(canvases[p], filename, Point(0, 24), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 0));
                        // writer << canvases[p];
                    }

                    imwrite(imgPath, canvases[p]);
                    pageUpdated[p] = false;

                    std::fill(vchUpdated.begin(), vchUpdated.end(), false);
                }
            }
        }

        if (dayChanged)
            createLog();
    }
};