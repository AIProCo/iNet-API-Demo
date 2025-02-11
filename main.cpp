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
#else
#include <filesystem>
#endif

#include <opencv2/core/core.hpp>

// core
#include "global.h"
#include "generator.h"
#include "videostreamer.hpp"

// util
#include "util.h"

#define DRAW_DETECTION_INFO false
#define DRAW_FIRE_DETECTION true
#define DRAW_FIRE_DETECTION_COUNTING true
#define DRAW_CNTLINE true
#define DRAW_CNTLINE_COUNTING true
#define DRAW_ZONE true
#define DRAW_ZONE_COUNTING true
#define DRAW_CC true
#define DRAW_CC_COUNTING true

using namespace std;
using namespace cv;
using namespace std::chrono;

void drawZones(Config &cfg, ODRecord &odRcd, Mat &img, int vchID, double alpha);
void drawBoxes(Config &cfg, ODRecord &odRcd, MinObj &minObj, Mat &img, vector<DetBox> &dboxes, int vchID,
               double alpha = 0.7);
void drawFD(Config &cfg, FDRecord &fdRcd, Mat &img, int vchID, float fdScoreThFire, float fdScoreThSmoke);
void drawCC(Config &cfg, CCRecord &ccRcd, Mat &density, Mat &img, int vchID);

// start engine
int main() {
    Config cfg;
    vector<CInfo> cInfos;  // channel information for each vchID

    try {
        if (!parseConfigAPI(cfg, cInfos)) {  // parse config.json
            cout << "parseConfigAPI: Parsing Error!\n";
            return -1;
        }
    } catch (const std::exception &e) {
        cout << e.what() << endl;
        return -1;
    }

    if (!initModel(cfg)) {
        cout << "initModel: Initialization of the solution failed!\n";
        return -1;
    }

    VideoStreamer streamer(cfg, cInfos);

    vector<unsigned int> frameCnts;
    frameCnts.resize(cfg.numChannels, 0);
    unsigned int frameLimit = cfg.frameLimit;  // number of frames to be processed

    steady_clock::time_point start, endOD, endFD, endCC;
    vector<int> delayODs, delayFDs, delayCCs;

    int vchID = 0;
    int minObjCnt = 0;
    while (1) {
        Mat frame;

        if (!streamer.read(frame, vchID)) {
            cout << "End of Videos!\n";
            break;
        }

        unsigned int &frameCnt = frameCnts[vchID];
        CInfo &cInfo = cInfos[vchID];

        start = steady_clock::now();

        // object detection and tracking
        vector<DetBox> dboxes;
        if (cfg.odChannels[vchID]) {
            int minObjSize = 0;  // set only when minObjs are deleted in DLL
            runModel(dboxes, minObjSize, cInfo, frame, vchID, frameCnt, cfg.odScoreTh);

            if (minObjSize > 0)
                minObjCnt++;
        }

        endOD = steady_clock::now();

        // fire classification
        if (cfg.fdChannels[vchID]) {
            runModelFD(cInfo.fdRcd, frame, vchID);
        }

        endFD = steady_clock::now();

        // crowd counting
        Mat density;
        if (cfg.ccChannels[vchID]) {
#ifndef _CPU_INFER
            runModelCC(density, cInfo.ccRcd, frame, vchID);
#else
            if (cInfo.ccRcd.ccZones.size() > 0) {
                cInfo.ccRcd.ccZones[0].setCanvas(frame);  // only one ccZone
                runModelCC(density, cInfo.ccRcd, frame, vchID);
            }
#endif
        }

        endCC = steady_clock::now();

        if (cfg.recording) {
            if (cfg.odChannels[vchID])
                drawBoxes(cfg, cInfo.odRcd, cInfo.minObj, frame, dboxes, vchID);

            if (cfg.fdChannels[vchID])
                drawFD(cfg, cInfo.fdRcd, frame, vchID, cfg.fdScoreThFire, cfg.fdScoreThSmoke);

            if (cfg.ccChannels[vchID])
                drawCC(cfg, cInfo.ccRcd, density, frame, vchID);

            streamer.write(frame, vchID);  // write a frame to the output video
        }

        int delay = duration_cast<milliseconds>(endCC - start).count();
        int delayOD = duration_cast<milliseconds>(endOD - start).count();
        int delayFD = duration_cast<milliseconds>(endFD - endOD).count();
        int delayCC = duration_cast<milliseconds>(endCC - endFD).count();

        cout << std::format(
            "[{}]Frame{:>4}> Inference Delay: {:>2}ms (OD: {:>2}ms, FD: {:>2}ms, CC: {:>2}ms), Small Objects: {}\n",
            vchID, frameCnt, delay, delayOD, delayFD, delayCC, minObjCnt);

        if (frameCnt > 10 && frameCnt < 100) {  // skip the start frames and limit the number of elements
            if (cfg.odChannels[vchID])
                delayODs.push_back(delayOD);

            if (cfg.fdChannels[vchID])
                delayFDs.push_back(delayFD);

            if (cfg.ccChannels[vchID])
                delayCCs.push_back(delayCC);
        }

        frameCnt++;
        if (frameLimit > 0 && frameCnt > frameLimit) {
            cout << std::format("\nBreak loop at frameCnt={:>4} and frameLimit={:>4}\n", frameCnt, frameLimit);
            break;
        }

        vchID++;
        if (vchID >= cfg.numChannels)
            vchID = 0;
    }

    if (delayODs.size() > 0 || delayFDs.size() > 0 || delayCCs.size() > 0) {
        float avgDelayOD = 0.0f, avgDelayFD = 0.0f, avgDelayCC = 0.0f;

        if (delayODs.size() > 0)
            avgDelayOD = accumulate(delayODs.begin(), delayODs.end(), 0) / delayODs.size();
        if (delayFDs.size() > 0)
            avgDelayFD = accumulate(delayFDs.begin(), delayFDs.end(), 0) / delayFDs.size();
        if (delayCCs.size() > 0)
            avgDelayCC = accumulate(delayCCs.begin(), delayCCs.end(), 0) / delayCCs.size();

        float avgDelay = avgDelayOD + avgDelayFD + avgDelayCC;
        cout << std::format("\nAverage Delay: {:>2}ms (OD: {:>2}ms, FD: {:>2}ms, CC: {:>2}ms)\n", avgDelay, avgDelayOD,
                            avgDelayFD, avgDelayCC);
    }

    if (cfg.recording) {
        cout << "\nOutput file(s):\n";
        for (auto &outFile : cfg.outputFiles)
            cout << std::format("  -{}\n", outFile);
    }

    streamer.destroy();  // destroy streamer
    destroyModel();      // destroy all models

    cout << "\nTerminate program!\n";

    return 0;
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

void drawBoxes(Config &cfg, ODRecord &odRcd, MinObj &minObj, Mat &img, vector<DetBox> &dboxes, int vchID,
               double alpha) {
    const string *objNames = cfg.odIDMapping.data();
    time_t now = time(NULL);

    vector<Rect> boxes;
    vector<Scalar> boxesColor;
    vector<bool> emphasizes;
    vector<vector<string>> boxTexts;

    for (auto &dbox : dboxes) {
        if (dbox.objID >= cfg.numClasses)
            continue;

        int label = dbox.objID;
        if (dbox.prob < cfg.odScoreTh)
            continue;  // should check scores are ordered. Otherwise, use continue

        Rect box(dbox.x, dbox.y, dbox.w, dbox.h);
        boxes.push_back(box);

        Scalar boxColor(50, 255, 255);
        vector<string> texts;

        bool isFemale;
        int probFemale;

        if (cfg.parEnable && dbox.patts.setCnt != -1) {
            PedAtts::getGenderAtt(dbox.patts, isFemale, probFemale);
            boxColor = isFemale ? Scalar(80, 80, 255) : Scalar(255, 80, 80);
        }

        int partitionIdx = (dbox.y + dbox.h) / (img.rows / 4);  //(dbox.y + dbox.h): 0 ~ H-1
        if (minObj.mode == MIN_OBJ_IN_DRAW) {
            if ((minObj.ths[partitionIdx] > 0) && (dbox.h * dbox.w < minObj.ths[partitionIdx]))
                boxColor = Scalar(230, 255, 20);  // ignored box;
        }

        // boxColor = Scalar(0, 255, 0); //for hsw

        if (DRAW_DETECTION_INFO) {
            // string objName = objNames[label] + "(" + to_string((int)(dbox.prob * 100 + 0.5)) + "%)";
            string objName = std::format("{}{}({:.1f}):{}({})", dbox.trackID, objNames[label], dbox.prob * 100 + 0.5,
                                         dbox.w * dbox.h, partitionIdx);
            // string objName = to_string(dbox.trackID) + objNames[label] + "(" + to_string((int)(dbox.prob * 100 +
            // 0.5)) + "%)";
            // string objName = to_string(dbox.trackID);

            // char buf[80];
            // tm *curTm = localtime(&dbox.inTime);
            // strftime(buf, sizeof(buf), "Time: %H:%M:%S", curTm);
            // string timeInfo = string(buf);

            texts.push_back(objName);
            // vector<string> texts{objName, timeInfo};

            if (0 && label == PERSON) {
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

        Vis::drawTextBlock(img, Point(18, 500), texts, 1, 2);
    }

    if (DRAW_CNTLINE) {
        for (CntLine &cntLine : odRcd.cntLines) {
            line(img, cntLine.pts[0], cntLine.pts[1], Scalar(50, 255, 50), 2, LINE_8);
        }
    }

    // draw couniting results
    if (DRAW_CNTLINE_COUNTING) {
        vector<string> texts = {"People Counting for Each Line"};

        for (CntLine &cntLine : odRcd.cntLines) {
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

        Vis::drawTextBlock(img, Point(18, 140), texts, 1, 2);
    }
}

void drawFD(Config &cfg, FDRecord &fdRcd, Mat &img, int vchID, float fdScoreThFire, float fdScoreThSmoke) {
    time_t now = time(NULL);
    int h = img.rows;
    int w = img.cols;
    string strFire = "X", strSmoke = "X";

    if (h < 720 || w < 1280)
        return;

    const int fx = w - 190, fy = 290;
    const vector<Point> ptsFire = {Point(fx + 0, fy + 54),  Point(fx + 24, fy + 0),  Point(fx + 45, fy + 48),
                                   Point(fx + 54, fy + 21), Point(fx + 63, fy + 54), Point(fx + 54, fy + 87),
                                   Point(fx + 15, fy + 87)};

    const int sx = w - 100, sy = 290;
    const vector<Point> ptsSmoke = {Point(sx + 0, sy + 51),  Point(sx + 0, sy + 30),  Point(sx + 21, sy + 0),
                                    Point(sx + 21, sy + 33), Point(sx + 63, sy + 42), Point(sx + 54, sy + 60),
                                    Point(sx + 36, sy + 87), Point(sx + 36, sy + 65)};

    if (fdRcd.fireProbs.back() > fdScoreThFire) {
        if (DRAW_FIRE_DETECTION) {
            /// draw canvas
            Mat fdIconRegion = img(Rect(Point(fx - 4, fy - 2), Point(fx + 63 + 4, fy + 87 + 2)));
            fdIconRegion -= Scalar(100, 100, 100);
            fillPoly(img, ptsFire, Scalar(0, 0, 255));
        }

        strFire = "O";
    }

    if (fdRcd.smokeProbs.back() > fdScoreThSmoke) {
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
    if (DRAW_CC) {
        if (cfg.boostMode) {
            if (!density.empty()) {
                vector<Mat> chans(3);

                split(img, chans);
                chans[2] += density;  // add to red channel
                merge(chans, img);
            }

            float alpha = 0.7f;
            int np[1] = {4};
            cv::Mat layer;

            for (CCZone &ccZone : ccRcd.ccZones) {
                if (layer.empty())
                    layer = img.clone();

                int z = ccZone.ccZoneID;
                const Scalar color(50, 50, 255);
                fillPoly(layer, {ccZone.pts}, color);
            }

            if (!layer.empty())
                cv::addWeighted(img, alpha, layer, 1 - alpha, 0, img);
        } else {
            for (CCZone &ccZone : ccRcd.ccZones) {
                const Scalar color(20, 20, 255);
                polylines(img, {ccZone.pts}, true, color, 2);
            }
        }
    }

    if (img.rows < 720 || img.cols < 1280)
        return;

    ////////////////////
    // draw couniting results
    if (DRAW_CC_COUNTING) {
        vector<string> ccTexts;
        ccTexts.push_back(string("Crowd Counting for Each CZone"));

#ifndef _CPU_INFER
        ccTexts.push_back(std::format("  Entire Area:{:>5}", ccRcd.ccNumFrames.back()));
#endif

        for (CCZone &ccZone : ccRcd.ccZones) {
            string text =
                // std::format(" -CZone {}: {:>4}", ccZone.ccZoneID+1, ccZone.ccNums.back());
                std::format("  CZone {}:{:>7}(L{})", ccZone.ccZoneID, ccZone.ccNums.back(), ccZone.ccLevel);
            ccTexts.push_back(text);
        }

        Vis::drawTextBlock(img, Point(img.cols - 555, 500), ccTexts, 1, 2);

        // for demo
        // vector<string> tmp0 = {string("CZone 1")};
        // Vis::drawTextBlock2(img, Point(800, 200), tmp0, 1, 2);
    }
}