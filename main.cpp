/*==============================================================================
* Copyright 2024 AIPro Inc.
* Author: Chun-Su Park (cspk@skku.edu)
=============================================================================*/
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

// core
#include "global.h"
#include "generator.h"

// util
#include "json.hpp"
#include "util.h"

#include "logger.h"
#include "CameraStreamer.hpp"

#define DRAW_DETECTION_INFO false
#define DRAW_FIRE_DETECTION true
#define DRAW_FIRE_DETECTION_COUNTING true
#define DRAW_CNTLINE true
#define DRAW_CNTLINE_COUNTING true
#define DRAW_ZONE true
#define DRAW_ZONE_COUNTING true
#define DRAW_CC true
#define DRAW_CC_COUNTING true

#define CFG_FILEPATH "inputs/config.json"
#define CC_MD_FILEPATH "inputs/aipro_cc_1_4_2.net"
#define CC_MD_CPU_FILEPATH "inputs/aipro_cc_1_4_2_cpu.nez"
#define OD_MD_FILEPATH "inputs/aipro_od_1_4.net"
#define OD_MD_CPU_FILEPATH "inputs/aipro_od_1_4_cpu.nez"
#define FD_MD_FILEPATH "inputs/aipro_fd_1_4_2.net"
#define FD_MD_CPU_FILEPATH "inputs/aipro_fd_1_4_2_cpu.nez"
#define PAR_MD_FILEPATH "inputs/aipro_par_1_4.net"
#define PAR_MD_CPU_FILEPATH "inputs/aipro_par_1_4_cpu.nez"

using namespace std;
using namespace cv;
using namespace std::filesystem;
using namespace std::chrono;

using json = nlohmann::json;

bool parseConfigAPI(Config &cfg, ODRecord &odRcd, FDRecord &fdRcd, CCRecord &ccRcd);
bool setODRecord(ODRecord &odRcd, vector<vector<int>> &cntLineParams, vector<vector<int>> &zoneParams);
bool setCCRecord(Config &cfg, CCRecord &ccRcd, vector<vector<int>> &ccZoneParams);
bool setFDRecord(Config &cfg, FDRecord &fdRcd);
void loadIS(Config &cfg, string txtPathInit, vector<vector<int>> &cntLineParams, vector<vector<int>> &zoneParams,
            vector<vector<int>> &ccZoneParams);
void drawZones(Config &cfg, ODRecord &odRcd, Mat &img, int vchID, double alpha);
void drawBoxes(Config &cfg, ODRecord &odRcd, Mat &img, vector<DetBox> &dboxes, int vchID, double alpha = 0.7);
void drawFD(Config &cfg, FDRecord &fdRcd, Mat &img, int vchID, float fdScoreTh);
void drawCC(Config &cfg, CCRecord &ccRcd, Mat &density, Mat &img, int vchID);

void doRunModelFD(Mat &frame, int vchID, uint frameCnt);
void doRunModelCC(Mat &density, Mat &frame, int vchID);

std::atomic<bool> ccTreadDone = false;
std::atomic<bool> fdTreadDone = false;

int main() {
    Config cfg;
    ODRecord odRcd;
    FDRecord fdRcd;
    CCRecord ccRcd;

    if (!parseConfigAPI(cfg, odRcd, fdRcd, ccRcd)) {
        cfg.lg("parseConfigAPI: Parsing Error!\n");
        return -1;
    }

    int numFDChannels = 0, numCCChannels = 0;
    for (int i = 0; i < cfg.numChannels; i++) {
        if (cfg.fdChannels[i])
            numFDChannels++;

        if (cfg.ccChannels[i])
            numCCChannels++;
    }

    int fdMinPeriod, ccMinPeriod;
    if (cfg.boostMode) {
        fdMinPeriod = 1;
        ccMinPeriod = 1;
    } else {
        fdMinPeriod = 10;
        ccMinPeriod = 17;
    }
    int fdPeriod = fdMinPeriod, ccPeriod = ccMinPeriod;

    std::function<void(std::string)> lg = cfg.lg;

    if (!initModel(cfg, odRcd, fdRcd, ccRcd)) {
        lg("initModel: Initialization of the solution failed!\n");
        return -1;
    }

    Logger logger(cfg, odRcd, fdRcd, ccRcd);
    CameraStreamer streamer(cfg, odRcd, fdRcd, ccRcd);

    unsigned int frameCnt = 0;                 // should be unsigned int
    unsigned int frameLimit = cfg.frameLimit;  // number of frames to be processed
    int vchID = 0, vchIDFD, vchIDCC;
    int odBatchSize = cfg.odBatchSize;  // batch size of object detection

    clock_t start, end;
    vector<float> infs0, infs1;
    int sleepPeriodMain = 0;
    vector<unsigned int> threadStartFDs(cfg.numChannels, 0);
    vector<unsigned int> threadStartCCs(cfg.numChannels, 0);
    thread *fdTh = NULL, *ccTh = NULL;

    vector<Mat> densityChPre(cfg.numChannels);
    Mat density;

    Mat frameFD(NET_HEIGHT_FD, NET_WIDTH_FD, CV_8UC3);
#ifndef _CPU_INFER
    Mat frameCC(NET_HEIGHT_CC, NET_WIDTH_CC, CV_8UC3);
#else
    Mat frameCC;
#endif

    while (1) {
        if (sleepPeriodMain > 0)
            Sleep(sleepPeriodMain);

        if (!streamer.empty()) {
            CMat cframe;
            streamer.try_pop(cframe);

            Mat frame = cframe.frame;
            vchID = cframe.vchID;  // fix odBatchSize to 1
            frameCnt = cframe.frameCnt;

            if (!logger.checkCmd(cfg, odRcd, fdRcd, ccRcd)) {
                lg("Break loop by logger.checkCmd\n");
                break;
            }

            system_clock::time_point now = system_clock::now();
            int ccNumFrame = 0;

            start = clock();

            if (cfg.fdChannels[vchID]) {
                unsigned int &threadStartFD = threadStartFDs[vchID];

                if (fdTh != NULL && fdTreadDone && vchID == vchIDFD) {
                    fdTreadDone = false;

                    fdTh->join();
                    fdTh = NULL;

                    bool periodChanged = true;
                    int threadGap = frameCnt - threadStartFD;
                    if (fdPeriod < (numFDChannels - 1) * threadGap)
                        ++fdPeriod;
                    else if (fdPeriod > numFDChannels * threadGap && fdPeriod > fdMinPeriod)
                        --fdPeriod;
                    else
                        periodChanged = false;

                    // periodChanged = true;
                    if (periodChanged)
                        lg(std::format("  [{}]Frame {}: FD thread Gap = {}({} - {}), period = {}\n", vchID, frameCnt,
                                       threadGap, frameCnt, threadStartFD, fdPeriod));

                    threadStartFD = frameCnt;  // for the following if-statement
                }

                if (threadStartFD + fdPeriod < frameCnt && fdTh == NULL) {
                    // lg(std::format("[{}]FD thread Start frameCnt={} > threadStartFD={} + fdPeriod={}\n", vchID,
                    // frameCnt, threadStartFD, fdPeriod));
                    resize(frame, frameFD, Size(cfg.fdNetWidth, cfg.fdNetHeight));
                    threadStartFD = frameCnt;
                    vchIDFD = vchID;
                    fdTh = new thread(doRunModelFD, ref(frameFD), vchID, frameCnt);
                }
            }

            if (cfg.ccChannels[vchID]) {
                unsigned int &threadStartCC = threadStartCCs[vchID];

                if (ccTh != NULL && ccTreadDone && vchID == vchIDCC) {
                    ccTreadDone = false;

                    ccTh->join();  //필요할까??
                    ccTh = NULL;

                    densityChPre[vchIDCC] = density;

                    bool periodChanged = true;
                    int threadGap = frameCnt - threadStartCC;
                    if (ccPeriod < (numCCChannels - 1) * threadGap)
                        ++ccPeriod;
                    else if (ccPeriod > numCCChannels * threadGap && ccPeriod > ccMinPeriod)
                        --ccPeriod;
                    else
                        periodChanged = false;

                    // periodChanged = true;
                    if (periodChanged)
                        lg(std::format("  [{}]Frame {}: CC thread Gap = {}({} - {}), period = {}\n", vchID, frameCnt,
                                       threadGap, frameCnt, threadStartCC, ccPeriod));

                    threadStartCC = frameCnt;  // for the following if-statement
                }

                if (threadStartCC + ccPeriod < frameCnt && ccTh == NULL) {
                    // lg(std::format("  [{}]CC thread Start frameCnt={} > threadStartCC={} + ccPeriod={}\n", vchID,
                    // frameCnt, threadStartCC, ccPeriod));
                    threadStartCC = frameCnt;
                    vchIDCC = vchID;
#ifndef _CPU_INFER
                    resize(frame, frameCC, Size(cfg.ccNetWidth, cfg.ccNetHeight));
                    ccTh = new thread(doRunModelCC, ref(density), ref(frameCC), vchID);
#else
                    for (CCZone &ccZone : ccRcd.ccZones) {
                        if (ccZone.vchID == vchID) {
                            ccZone.setCanvas(frame);
                            ccTh = new thread(doRunModelCC, ref(density), ref(frameCC), vchID);
                            break;
                        }
                    }
#endif
                }
            }

            vector<DetBox> dboxes;
            if (cfg.odChannels[vchID])
                runModel(dboxes, frame, vchID, frameCnt, cfg.odScoreTh);

            bool needToDraw = logger.needToDraw(vchID);
            if (needToDraw || cfg.recording) {
                if (cfg.odChannels[vchID])
                    drawBoxes(cfg, odRcd, frame, dboxes, vchID);

                if (cfg.fdChannels[vchID])
                    drawFD(cfg, fdRcd, frame, vchID, cfg.fdScoreTh);

                if (cfg.ccChannels[vchID])
                    drawCC(cfg, ccRcd, densityChPre[vchID], frame, vchID);
            }

            end = clock();

            if (cfg.recording)
                (streamer[vchID]) << frame;  // write a frame

            logger.writeData(cfg, odRcd, fdRcd, ccRcd, frame, frameCnt, vchID, now);

            float inf0 = (end - start) / odBatchSize;

            if (frameCnt % 20 == 0)
                lg(std::format("[{}]Frame{:>4}>  SP({:>2}, {:>3}), Gap(FD: {}, CC: {}),  Buf: {},  Inf: {}ms\n", vchID,
                               frameCnt, sleepPeriodMain, streamer.sleepPeriod, fdPeriod, ccPeriod,
                               streamer.unsafe_size(), inf0));

            if (frameCnt > 10 && frameCnt < 500)  // skip the start frames and limit the number of elements
                infs0.push_back(inf0);

            if (streamer.unsafe_size() > 0)
                sleepPeriodMain = 0;
        } else {
            if (sleepPeriodMain < 10)
                sleepPeriodMain += 2;
        }

        if (frameLimit > 0 && frameCnt > frameLimit) {
            lg(std::format("\nBreak loop at frameCnt={:>4} and frameLimit={:>4}\n", frameCnt, frameLimit));
            break;
        }
    }

    if (fdTh)
        fdTh->join();

    if (ccTh)
        ccTh->join();

    destroyModel();      // destroy all models
    streamer.destory();  // destroy streamer

    if (infs0.size() > 1) {
        float avgInf0 = accumulate(infs0.begin(), infs0.end(), 0) / infs0.size();
        lg(std::format("\nAverage Inference Time: {}ms\n", avgInf0));
    }

    if (cfg.recording) {
        lg("\nOutput file(s):\n");
        for (auto &outFile : cfg.outputFiles)
            lg(std::format("  -{}\n", outFile));
    }

    lg("\nTerminate program!\n");

    return 0;
}

void doRunModelFD(Mat &frame, int vchID, uint frameCnt) {
    runModelFD(frame, vchID, frameCnt);
    fdTreadDone = true;
}

void doRunModelCC(Mat &density, Mat &frame, int vchID) {
    runModelCC(density, frame, vchID);
    ccTreadDone = true;
}

void drawZones(Config &cfg, ODRecord &odRcd, Mat &img, int vchID, double alpha) {
    if (cfg.boostMode) {
        int np[1] = {4};
        cv::Mat layer;

        for (Zone &zone : odRcd.zones) {
            if (zone.vchID == vchID) {
                if (layer.empty())
                    layer = img.clone();

                int z = zone.zoneID;
                const Scalar color(255, 50, 50);
                fillPoly(layer, {zone.pts}, color);
            }
        }

        if (!layer.empty())
            cv::addWeighted(img, alpha, layer, 1 - alpha, 0, img);
    } else {
        for (Zone &zone : odRcd.zones) {
            if (zone.vchID == vchID) {
                const Scalar color(255, 20, 20);
                polylines(img, {zone.pts}, true, color, 2);
            }
        }
    }
}

void drawBoxes(Config &cfg, ODRecord &odRcd, Mat &img, vector<DetBox> &dboxes, int vchID, double alpha) {
    const string *objNames = cfg.odIDMapping.data();
    time_t now = time(NULL);

    vector<Rect> boxes;
    vector<Scalar> boxesColor;
    vector<bool> emphasizes;
    vector<vector<string>> boxTexts;

    for (auto &dbox : dboxes) {
        if (dbox.objID >= cfg.numClasses)
            continue;

        Rect box(dbox.x, dbox.y, dbox.w, dbox.h);
        boxes.push_back(box);

        int label = dbox.objID;
        if (dbox.prob < cfg.odScoreTh)
            continue;  // should check scores are ordered. Otherwise, use continue

        Scalar boxColor(50, 255, 255);
        vector<string> texts;

        bool isFemale;
        int probFemale;

        if (cfg.parEnable && dbox.patts.setCnt != -1) {
            PedAtts::getGenderAtt(dbox.patts, isFemale, probFemale);
            boxColor = isFemale ? Scalar(80, 80, 255) : Scalar(255, 80, 80);
        }

        if (DRAW_DETECTION_INFO) {
            // string objName = objNames[label] + "(" + to_string((int)(dbox.prob * 100 + 0.5)) + "%)";
            string objName =
                to_string(dbox.trackID) + objNames[label] + "(" + to_string((int)(dbox.prob * 100 + 0.5)) + "%)";
            // string objName = to_string(dbox.trackID);

            // char buf[80];
            // tm *curTm = localtime(&dbox.inTime);
            // strftime(buf, sizeof(buf), "Time: %H:%M:%S", curTm);
            // string timeInfo = string(buf);

            texts.push_back(objName);
            // vector<string> texts{objName, timeInfo};

            if (label == PERSON) {
                string trkInfo;
                int period = now - dbox.inTime;
                if (period < cfg.longLastingObjTh) {  // no action
                    trkInfo = "ET: " + to_string(period) + " (" + to_string((int)dbox.distVar) + ")";
                } else {                              // action (Sleep or Hang around)
                    if (dbox.distVar < cfg.noMoveTh)  // Sleep event
                        trkInfo = "ET: " + to_string(period) + ", No movement(" + to_string((int)dbox.distVar) + ")";
                    else  // Hang around event
                        trkInfo = "ET: " + to_string(period) + ", Hang around(" + to_string((int)dbox.distVar) + ")";
                }
                texts.push_back(trkInfo);

                if (cfg.parEnable && dbox.patts.setCnt != -1) {
                    string genderInfo, ageGroupInfo;
                    int ageGroup, probAgeGroup;

                    genderInfo = "Gen: " + string((isFemale ? "F" : "M")) + " (" + to_string(probFemale) + "%)" +
                                 to_string(dbox.patts.setCnt);
                    texts.push_back(genderInfo);

                    PedAtts::getAgeGroupAtt(dbox.patts, ageGroup, probAgeGroup);
                    ageGroupInfo =
                        "Age: " +
                        string(ageGroup == CHILD_GROUP ? "child" : (ageGroup == ADULT_GROUP ? "adult" : "elder")) +
                        " (" + to_string(probAgeGroup) + "%)";
                    texts.push_back(ageGroupInfo);
                }
            }
        }

        boxesColor.push_back(boxColor);
        boxTexts.push_back(texts);

        if ((DRAW_CNTLINE && (dbox.justCountedLine > 0)) || (DRAW_ZONE && (dbox.justCountedZone > 0)))
            emphasizes.push_back(true);
        else
            emphasizes.push_back(false);
    }

    Vis::drawBoxes(img, boxes, boxesColor, boxTexts, emphasizes);

    if (DRAW_ZONE)
        drawZones(cfg, odRcd, img, vchID, alpha);

    // draw par results
    if (DRAW_ZONE_COUNTING) {
        vector<string> texts = {"People Counting for Each Zone"};

        for (Zone &zone : odRcd.zones) {
            if (vchID == zone.vchID) {
                int curMTotal, curFTotal;
                curMTotal = zone.curPeople[0][0] + zone.curPeople[0][1] + zone.curPeople[0][2];
                curFTotal = zone.curPeople[1][0] + zone.curPeople[1][1] + zone.curPeople[1][2];

                string title = " Zone " + to_string(zone.zoneID);
                string cur = "   Cur> M: " + to_string(curMTotal) + "(" + to_string(zone.curPeople[0][0]) + ", " +
                             to_string(zone.curPeople[0][1]) + ", " + to_string(zone.curPeople[0][2]) + "), " +
                             " F: " + to_string(curFTotal) + "(" + to_string(zone.curPeople[1][0]) + ", " +
                             to_string(zone.curPeople[1][1]) + ", " + to_string(zone.curPeople[1][2]) + ")";

                int hitMTotal, hitFTotal;
                hitMTotal = zone.hitMap[0][0] + zone.hitMap[0][1] + zone.hitMap[0][2];
                hitFTotal = zone.hitMap[1][0] + zone.hitMap[1][1] + zone.hitMap[1][2];

                string hit = "   Hit> M: " + to_string(hitMTotal) + "(" + to_string(zone.hitMap[0][0]) + ", " +
                             to_string(zone.hitMap[0][1]) + ", " + to_string(zone.hitMap[0][2]) + "), " +
                             " F: " + to_string(hitFTotal) + "(" + to_string(zone.hitMap[1][0]) + ", " +
                             to_string(zone.hitMap[1][1]) + ", " + to_string(zone.hitMap[1][2]) + ")";

                texts.push_back(title);
                texts.push_back(cur);
                texts.push_back(hit);
            }
        }

        Vis::drawTextBlock(img, Point(18, 500), texts, 1, 2);
    }

    if (DRAW_CNTLINE) {
        for (CntLine &cntLine : odRcd.cntLines) {
            if (vchID == cntLine.vchID)
                line(img, cntLine.pts[0], cntLine.pts[1], Scalar(50, 255, 50), 2, LINE_8);
        }
    }

    // draw couniting results
    if (DRAW_CNTLINE_COUNTING) {
        vector<string> texts = {"People Counting for Each Line"};

        for (CntLine &cntLine : odRcd.cntLines) {
            if (vchID == cntLine.vchID) {
                int upMTotal, upFTotal;
                upMTotal = cntLine.totalUL[0][0] + cntLine.totalUL[0][1] + cntLine.totalUL[0][2];
                upFTotal = cntLine.totalUL[1][0] + cntLine.totalUL[1][1] + cntLine.totalUL[1][2];

                string title = " Counting Line " + to_string(cntLine.clineID);
                string up = "   U/L> M: " + to_string(upMTotal) + "(" + to_string(cntLine.totalUL[0][0]) + ", " +
                            to_string(cntLine.totalUL[0][1]) + ", " + to_string(cntLine.totalUL[0][2]) + "), " +
                            " F: " + to_string(upFTotal) + "(" + to_string(cntLine.totalUL[1][0]) + ", " +
                            to_string(cntLine.totalUL[1][1]) + ", " + to_string(cntLine.totalUL[1][2]) + ")";

                int dwMTotal, dwFTotal;
                dwMTotal = cntLine.totalDR[0][0] + cntLine.totalDR[0][1] + cntLine.totalDR[0][2];
                dwFTotal = cntLine.totalDR[1][0] + cntLine.totalDR[1][1] + cntLine.totalDR[1][2];

                string dw = "   D/R> M: " + to_string(dwMTotal) + "(" + to_string(cntLine.totalDR[0][0]) + ", " +
                            to_string(cntLine.totalDR[0][1]) + ", " + to_string(cntLine.totalDR[0][2]) + "), " +
                            " F: " + to_string(dwFTotal) + "(" + to_string(cntLine.totalDR[1][0]) + ", " +
                            to_string(cntLine.totalDR[1][1]) + ", " + to_string(cntLine.totalDR[1][2]) + ")";

                texts.push_back(title);
                texts.push_back(up);
                texts.push_back(dw);
            }
        }

        Vis::drawTextBlock(img, Point(18, 140), texts, 1, 2);
    }
}

void drawFD(Config &cfg, FDRecord &fdRcd, Mat &img, int vchID, float fdScoreTh) {
    time_t now = time(NULL);
    int h = img.rows;
    int w = img.cols;
    string strFire = "X", strSmoke = "X";

    if (h < 380 || w < 500)
        return;

    const int fx = w - 190, fy = 290;
    const vector<Point> ptsFire = {Point(fx + 0, fy + 54),  Point(fx + 24, fy + 0),  Point(fx + 45, fy + 48),
                                   Point(fx + 54, fy + 21), Point(fx + 63, fy + 54), Point(fx + 54, fy + 87),
                                   Point(fx + 15, fy + 87)};

    const int sx = w - 100, sy = 290;
    const vector<Point> ptsSmoke = {Point(sx + 0, sy + 51),  Point(sx + 0, sy + 30),  Point(sx + 21, sy + 0),
                                    Point(sx + 21, sy + 33), Point(sx + 63, sy + 42), Point(sx + 54, sy + 60),
                                    Point(sx + 36, sy + 87), Point(sx + 36, sy + 65)};

    if (fdRcd.fireProbsMul[vchID].back() > fdScoreTh) {
        if (DRAW_FIRE_DETECTION) {
            /// draw canvas
            Mat fdIconRegion = img(Rect(Point(fx - 4, fy - 2), Point(fx + 63 + 4, fy + 87 + 2)));
            fdIconRegion -= Scalar(100, 100, 100);
            fillPoly(img, ptsFire, Scalar(0, 0, 255));
        }

        strFire = "O";
    }

    if (fdRcd.smokeProbsMul[vchID].back() > fdScoreTh) {
        if (DRAW_FIRE_DETECTION) {
            Mat smokeIconRegion = img(Rect(Point(sx - 4, sy - 2), Point(sx + 63 + 4, sy + 87 + 2)));
            smokeIconRegion -= Scalar(100, 100, 100);
            fillPoly(img, ptsSmoke, Scalar(200, 200, 200));
        }

        strSmoke = "O";
    }

    if (DRAW_FIRE_DETECTION_COUNTING) {
        string fdText = "Event> Fire: " + strFire + ", Smoke: " + strSmoke;
        Vis::drawTextBlockFD(img, fdRcd, vchID, 140, fdText, 1, 2);

        // if (cfg.boostMode && (strSmoke == "O" || strFire == "O"))
        //    rectangle(img, Rect(0, 0, img.cols, img.rows), Scalar(0, 0, 255), 4);
    }
}

void drawCC(Config &cfg, CCRecord &ccRcd, Mat &density, Mat &img, int vchID) {
    if (img.rows < 380 || img.cols < 500)
        return;

    if (DRAW_CC) {
        if (cfg.boostMode) {
            if (!density.empty()) {
                vector<Mat> chans(3);

                split(img, chans);
                chans[2] += density;  // add to red channel
                merge(chans, img);
            }

            float alpha = 0.7;
            int np[1] = {4};
            cv::Mat layer;

            for (CCZone &ccZone : ccRcd.ccZones) {
                if (ccZone.vchID == vchID) {
                    if (layer.empty())
                        layer = img.clone();

                    int z = ccZone.ccZoneID;
                    const Scalar color(50, 50, 255);
                    fillPoly(layer, {ccZone.pts}, color);
                }
            }

            if (!layer.empty())
                cv::addWeighted(img, alpha, layer, 1 - alpha, 0, img);
        } else {
            for (CCZone &ccZone : ccRcd.ccZones) {
                if (ccZone.vchID == vchID) {
                    const Scalar color(20, 20, 255);
                    polylines(img, {ccZone.pts}, true, color, 2);
                }
            }
        }
    }

    ////////////////////
    // draw couniting results
    if (DRAW_CC_COUNTING) {
        vector<string> ccTexts;
        ccTexts.push_back(string("Crowd Counting for Each CZone"));

#ifndef _CPU_INFER
        ccTexts.push_back(std::format("  Entire Area:{:>5}", ccRcd.ccNumFrames[vchID].back()));
#endif

        for (CCZone &ccZone : ccRcd.ccZones) {
            if (ccZone.vchID == vchID) {
                string text =
                    // std::format(" -CZone {}: {:>4}", ccZone.ccZoneID+1, ccZone.ccNums.back());
                    std::format("  CZone {}:{:>7}(L{})", ccZone.ccZoneID, ccZone.ccNums.back(), ccZone.ccLevel);
                ccTexts.push_back(text);
            }
        }

        Vis::drawTextBlock(img, Point(img.cols - 555, 500), ccTexts, 1, 2);

        // for demo
        // vector<string> tmp0 = {string("CZone 1")};
        // Vis::drawTextBlock2(img, Point(800, 200), tmp0, 1, 2);
    }
}

bool parseConfigAPI(Config &cfg, ODRecord &odRcd, FDRecord &fdRcd, CCRecord &ccRcd) {
    string jsonCfgFile = CFG_FILEPATH;
    std::ifstream cfgFile(jsonCfgFile);
    json js;
    cfgFile >> js;

    // logger
    cfg.lg = Logger::writeLog;
    cfg.lg("Start parsing config....\n");

    // apikey, gpu_id
    cfg.frameLimit = js["global"]["frame_limit"];
    cfg.key = js["global"]["apikey"];
    cfg.recording = js["global"]["recording"];
    cfg.debugMode = js["global"]["debug_mode"];
    cfg.boostMode = js["global"]["boost_mode"];
    cfg.igpuEnable = js["global"]["igpu_enable"];
    cfg.logEnable = true;  // fixed

    int parsingMode = js["global"]["parsing_mode"];

    string filenameCam = "cam.json";
    string jsonPathCam = string(CONFIG_PATH) + "/" + filenameCam;

    if ((parsingMode == 1 || parsingMode == 3) && std::filesystem::exists(jsonPathCam)) {
        std::ifstream fileCam(jsonPathCam);
        json jsCam;
        fileCam >> jsCam;

        cfg.numChannels = jsCam["num_channels"];

        for (int ch = 0; ch < cfg.numChannels; ch++) {
            string c = to_string(ch);
            cfg.inputFiles.push_back(jsCam[c]["all"]);
            cfg.odChannels.push_back(jsCam[c]["od_enable"]);
            cfg.fdChannels.push_back(jsCam[c]["fd_enable"]);
            cfg.ccChannels.push_back(jsCam[c]["cc_enable"]);
        }

        if (cfg.recording) {
            for (int ch = 0; ch < cfg.numChannels; ch++) {
                string filepath = string(VIDEO_OUT_PATH) + "/out" + to_string(ch) + ".mp4";
                cfg.outputFiles.push_back(filepath);
            }
        }

        if (cfg.outputFiles.size() > cfg.numChannels)
            cfg.outputFiles.resize(cfg.numChannels);
    } else {
        // read the list of filepaths
        cfg.inputFiles = js["global"]["input_files"].get<vector<string>>();
        cfg.outputFiles = js["global"]["output_files"].get<vector<string>>();

        if (cfg.inputFiles.size() != cfg.outputFiles.size()) {
            cout << "input_files and output_files should be the same size!!";
            return false;
        }

        cfg.numChannels = cfg.inputFiles.size();
        cfg.odChannels = js["global"]["od_channels"].get<vector<bool>>();
        cfg.fdChannels = js["global"]["fd_channels"].get<vector<bool>>();
        cfg.ccChannels = js["global"]["cc_channels"].get<vector<bool>>();

        if (cfg.odChannels.size() < cfg.numChannels) {
            for (int i = cfg.odChannels.size(); i < cfg.numChannels; i++)
                cfg.odChannels.push_back(false);
        } else if (cfg.odChannels.size() > cfg.numChannels) {
            cfg.odChannels.resize(cfg.numChannels);
        }

        if (cfg.fdChannels.size() < cfg.numChannels) {
            for (int i = cfg.fdChannels.size(); i < cfg.numChannels; i++)
                cfg.fdChannels.push_back(false);
        } else if (cfg.fdChannels.size() > cfg.numChannels) {
            cfg.fdChannels.resize(cfg.numChannels);
        }

        if (cfg.ccChannels.size() < cfg.numChannels) {
            for (int i = cfg.ccChannels.size(); i < cfg.numChannels; i++)
                cfg.ccChannels.push_back(false);
        } else if (cfg.ccChannels.size() > cfg.numChannels) {
            cfg.ccChannels.resize(cfg.numChannels);
        }
    }

    cfg.maxBufferSize = cfg.numChannels * 4;
    cfg.vchStates.resize(cfg.numChannels, 0);
    cfg.frameWidths.resize(cfg.numChannels, 0);
    cfg.frameHeights.resize(cfg.numChannels, 0);
    cfg.fpss.resize(cfg.numChannels, 0);

    // od config
    cfg.odEnable = js["od"]["enable"];
    cfg.odNetWidth = NET_WIDTH_OD;    // fixed
    cfg.odNetHeight = NET_HEIGHT_OD;  // fixed
    cfg.odScaleFactors.resize(cfg.numChannels, 0);
    cfg.odScaleFactorsInv.resize(cfg.numChannels, 0);
    cfg.odBatchSize = 1;  // fixed
    cfg.odScoreTh = js["od"]["score_th"];
    cfg.odIDMapping = {"person"};
    // cfg.odIDMapping = {"person", "bycle", "car", "motorcycle", "airplane", "bus", "train", "truck"};
    cfg.numClasses = cfg.odIDMapping.size();

    // fd config
    cfg.fdEnable = js["fd"]["enable"];
    cfg.fdNetWidth = NET_WIDTH_FD;    // fixed
    cfg.fdNetHeight = NET_HEIGHT_FD;  // fixed
    cfg.fdScaleFactors.resize(cfg.numChannels, 0);
    cfg.fdScaleFactorsInv.resize(cfg.numChannels, 0);
    cfg.fdBatchSize = 1;  // fixed(but support from 1 to 4)
    cfg.fdScoreTh = js["fd"]["score_th"];
    cfg.fdIDMapping = {"both", "fire", "none", "smoke"};  // fixed (should be the same as fd modes in global.h)
    // cfg.fdIDMapping = {"smoke", "fire"};  // fixed
    cfg.fdNumClasses = NUM_FD_CLASSES;
    cfg.fdWindowSize = 32;
    cfg.fdPeriod = 16;  // check period in the idle mode

    // cc config
    cfg.ccEnable = js["cc"]["enable"];
    cfg.ccNetWidth = NET_WIDTH_CC;    // fixed
    cfg.ccNetHeight = NET_HEIGHT_CC;  // fixed
    cfg.ccScaleFactors.resize(cfg.numChannels, 0);
    cfg.ccScaleFactorsInv.resize(cfg.numChannels, 0);
    cfg.ccWindowSize = 32;
    cfg.ccPeriod = 16;  // check period in the idle mode

    // tracking
    cfg.longLastingObjTh = 300;
    cfg.noMoveTh = 3.0f;
    cfg.lineEmphasizePeroid = 15;

    // par config
    cfg.parEnable = js["par"]["enable"];
    cfg.parBatchSize = 1;  // fixed
    cfg.parIDMapping = {"gender", "child", "adult", "elder"};
    cfg.numAtts = NUM_ATTRIBUTES;
    cfg.attUpdatePeriod = 15;

    // counting
    cfg.debouncingTh = 20;

#ifndef _CPU_INFER
    cfg.lg("Do inference using GPU\n");

    cfg.odModelFile = OD_MD_FILEPATH;
    cfg.fdModelFile = FD_MD_FILEPATH;
    cfg.parModelFile = PAR_MD_FILEPATH;
    cfg.ccModelFile = CC_MD_FILEPATH;
    cfg.parLightMode = true;
#else
    cfg.lg("Do inference using CPU\n");

    cfg.odModelFile = OD_MD_CPU_FILEPATH;
    cfg.fdModelFile = FD_MD_CPU_FILEPATH;
    cfg.parModelFile = PAR_MD_CPU_FILEPATH;
    cfg.ccModelFile = CC_MD_CPU_FILEPATH;
    cfg.parLightMode = true;
#endif
    cfg.lg(std::format("\nModel Configure:\n  -OD & Tracking: {}\n  -FD: {}\n  -PAR: {}\n  -CC: {}\n\n", cfg.odEnable,
                       cfg.fdEnable, cfg.parEnable, cfg.ccEnable));

    if (cfg.parLightMode)
        cfg.lineEmphasizePeroid = min(cfg.lineEmphasizePeroid, cfg.attUpdatePeriod);

    // Enforce channel characteristics based on whether each model is enabled or not
    for (int ch = 0; ch < cfg.numChannels; ch++) {
        if (!cfg.odEnable)
            cfg.odChannels[ch] = false;

        if (!cfg.fdEnable)
            cfg.fdChannels[ch] = false;

        if (!cfg.ccEnable)
            cfg.ccChannels[ch] = false;
    }

    vector<vector<int>> cntLineParams, zoneParams, ccZoneParams;
    string filenameInit = "is.txt";
    string txtPathInit = string(CONFIG_PATH) + "/" + filenameInit;

    if ((parsingMode == 2 || parsingMode == 3) && std::filesystem::exists(txtPathInit)) {
        cfg.lg("load is.txt for IS\n");
        loadIS(cfg, txtPathInit, cntLineParams, zoneParams, ccZoneParams);
    } else {
        cfg.lg("load config.json for IS\n");
        // cntline
        cntLineParams = js["line"]["param"].get<vector<vector<int>>>();

        // zones
        zoneParams = js["zone"]["param"].get<vector<vector<int>>>();

        // Crowd counting
        ccZoneParams = js["cc_zone"]["param"].get<vector<vector<int>>>();
    }

    if (!setODRecord(odRcd, cntLineParams, zoneParams)) {
        cfg.lg("setODRecord Error!\n");
        return false;
    }

    if (!setCCRecord(cfg, ccRcd, ccZoneParams)) {
        cfg.lg("setCCRecord Error!\n");
        return false;
    }

    if (!setFDRecord(cfg, fdRcd)) {
        cfg.lg("setFDRecord Error!\n");
        return false;
    }

    cfg.lg("End parsing config....\n\n");
    return true;
}

bool setFDRecord(Config &cfg, FDRecord &fdRcd) {
    if (cfg.fdEnable) {
        fdRcd.fireProbsMul.resize(cfg.numChannels);
        fdRcd.smokeProbsMul.resize(cfg.numChannels);

        for (int i = 0; i < cfg.numChannels; i++) {
            fdRcd.fireProbsMul[i].resize(cfg.fdWindowSize, 0.0f);
            fdRcd.smokeProbsMul[i].resize(cfg.fdWindowSize, 0.0f);
        }

        fdRcd.fireEvents.resize(cfg.numChannels, 0);
        fdRcd.smokeEvents.resize(cfg.numChannels, 0);
        fdRcd.afterFireEvents.resize(cfg.numChannels, 0);
    }

    return true;
}

bool setODRecord(ODRecord &odRcd, vector<vector<int>> &cntLineParams, vector<vector<int>> &zoneParams) {
    odRcd.cntLines.clear();
    odRcd.zones.clear();

    for (auto &cntLineParam : cntLineParams) {
        CntLine cntLine;

        cntLine.enabled = true;
        cntLine.preTotal = 0;
        cntLine.clineID = cntLineParam[0];
        cntLine.vchID = cntLineParam[1];
        cntLine.isMode = cntLineParam[2];
        cntLine.pts[0].x = cntLineParam[3];
        cntLine.pts[0].y = cntLineParam[4];
        cntLine.pts[1].x = cntLineParam[5];
        cntLine.pts[1].y = cntLineParam[6];

        if (abs(cntLine.pts[0].x - cntLine.pts[1].x) > abs(cntLine.pts[0].y - cntLine.pts[1].y))
            cntLine.direction = 0;  // horizontal line -> use delta y and count U and D
        else
            cntLine.direction = 1;  // vertical line -> use delta x and count L and R

        for (int g = 0; g < NUM_GENDERS; g++) {         // number of genders
            for (int a = 0; a < NUM_AGE_GROUPS; a++) {  // number of age groups
                cntLine.totalUL[g][a] = 0;
                cntLine.totalDR[g][a] = 0;
            }
        }

        odRcd.cntLines.push_back(cntLine);
    }

    if (zoneParams.size() > 100) {
        cout << "The number of zones should not exceed 100.\n";
        return false;
    }

    for (auto &zoneParam : zoneParams) {
        Zone zone;

        zone.enabled = true;
        zone.preTotal = 0;
        zone.zoneID = zoneParam[0];
        zone.vchID = zoneParam[1];
        zone.isMode = zoneParam[2];

        for (int i = 0; i < 4; i++) {
            Point pt;
            pt.x = zoneParam[2 * i + 3];
            pt.y = zoneParam[2 * i + 4];
            zone.pts.push_back(pt);
        }

        for (int g = 0; g < NUM_GENDERS; g++) {         // number of genders
            for (int a = 0; a < NUM_AGE_GROUPS; a++) {  // number of age groups
                zone.curPeople[g][a] = 0;
                zone.hitMap[g][a] = 0;
            }
        }

        odRcd.zones.push_back(zone);
    }

    return true;
}

bool setCCRecord(Config &cfg, CCRecord &ccRcd, vector<vector<int>> &ccZoneParams) {
    ccRcd.ccZones.clear();

    for (auto &ccZoneParam : ccZoneParams) {
        CCZone ccZone;

        ccZone.enabled = true;
        ccZone.ccLevel = 0;
        ccZone.preCCLevel = 0;

        ccZone.ccZoneID = ccZoneParam[0];
        ccZone.vchID = ccZoneParam[1];

        bool isFirst = true;

        for (auto &item : ccRcd.ccZones) {  // Store only one ccZone for each vchID
#ifndef _CPU_INFER
            if (item.vchID == ccZone.vchID && item.ccZoneID == ccZone.ccZoneID) {
#else
            if (item.vchID == ccZone.vchID) {  // only one ccZone(cpu mode)
#endif
                cfg.lg(std::format("Duplicated ccZone: {} {}\n", ccZone.vchID, ccZone.ccZoneID));
                isFirst = false;
                break;
            }
        }

        if (!isFirst)
            continue;

        for (int i = 0; i < 4; i++) {
            Point pt;
            pt.x = ccZoneParam[2 * i + 2];
            pt.y = ccZoneParam[2 * i + 3];
            ccZone.pts.push_back(pt);
        }

        ccZone.ccLevelThs[0] = ccZoneParam[10];
        ccZone.ccLevelThs[1] = ccZoneParam[11];
        ccZone.ccLevelThs[2] = ccZoneParam[12];

        if (ccZone.ccLevelThs[0] > ccZone.ccLevelThs[1] || ccZone.ccLevelThs[1] > ccZone.ccLevelThs[2]) {
            cfg.lg("ccLevelThs should be ordered.\n");
            return false;
        }

        ccZone.ccNums.resize(cfg.ccWindowSize, 0);

        ccZone.maxCC = 0;
        ccZone.maxCCDay = 0;

        for (int i = 0; i < NUM_CC_LEVELS - 1; i++) {
            ccZone.accCCLevels[i] = 0;
            ccZone.accCCLevelsDay[i] = 0;
        }

        // init mask with empty Mat. This is generated in CrowdCounter::runModel.
        ccZone.mask = cv::Mat();
#ifdef _CPU_INFER
        ccZone.canvas = cv::Mat();
        ccZone.roiCanvas = cv::Mat();
#endif
        ccRcd.ccZones.push_back(ccZone);
    }

    ccRcd.ccNumFrames.resize(cfg.numChannels);

    for (int i = 0; i < cfg.numChannels; i++)
        ccRcd.ccNumFrames[i].resize(cfg.ccWindowSize, 0.0f);

    return true;
}

void loadIS(Config &cfg, string txtPathInit, vector<vector<int>> &cntLineParams, vector<vector<int>> &zoneParams,
            vector<vector<int>> &ccZoneParams) {
    int elementsMinusOnes[3] = {7, 11, 13};  // cntLineParmas-1, zoneParams-1, ccZoneParams-1 (in is.txt)

    ifstream init(txtPathInit);

    if (init.is_open()) {
        int typeID;

        while (init >> typeID) {
            int data;
            if (typeID == 3) {  // read scoreThs
                init >> data;
                cfg.odScoreTh = (data / 1000.0f);

                init >> data;
                cfg.fdScoreTh = (data / 1000.0f);
            } else {  // read IS
                vector<int> line;
                int elementsMinusOne = elementsMinusOnes[typeID];

                for (int i = 0; i < elementsMinusOne; i++) {
                    init >> data;
                    line.push_back(data);
                }

                switch (typeID) {
                    case 0:
                        cntLineParams.push_back(line);
                        break;
                    case 1:
                        zoneParams.push_back(line);
                        break;
                    case 2:
                        ccZoneParams.push_back(line);
                        break;
                    default:
                        cfg.lg(std::format("typeID Error in loadIS: {}\n", typeID));
                        break;
                }
            }
        }
    }
}