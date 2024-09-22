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
#include <thread>

#ifdef _WIN32
#include <windows.h>
#else
#include <filesystem>
#endif

#include <opencv2/core/core.hpp>

// core
#include "global.h"
#include "generator.h"

// util
#include "util.h"

#define DRAW_DETECTION_INFO true
#define DRAW_FIRE_DETECTION true
#define DRAW_FIRE_DETECTION_COUNTING true
#define DRAW_CNTLINE false
#define DRAW_CNTLINE_COUNTING false
#define DRAW_ZONE false
#define DRAW_ZONE_COUNTING false
#define DRAW_CC true
#define DRAW_CC_COUNTING true

using namespace std;
using namespace cv;
using namespace std::filesystem;
using namespace std::chrono;

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

    setWriteLogger(cfg);  // set cfg.lg. This should be called before parseConfigAPI()
    std::function<void(std::string)> lg = cfg.lg;

    if (!parseConfigAPI(cfg, odRcd, fdRcd, ccRcd)) {  // parse config.json, cam.json, and is.txt
        lg("parseConfigAPI: Parsing Error!\n");
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
    if (cfg.boostMode) {  // cfg.boostMode
        fdMinPeriod = 1, ccMinPeriod = 1;
    } else {
        fdMinPeriod = 10, ccMinPeriod = 17;
    }
    int fdPeriod = fdMinPeriod, ccPeriod = ccMinPeriod;

    if (!initModel(cfg, odRcd, fdRcd, ccRcd)) {
        lg("initModel: Initialization of the solution failed!\n");
        return -1;
    }

    initLogger(cfg, odRcd, fdRcd, ccRcd);  // should be called before initStreamer
    initStreamer(cfg, odRcd, fdRcd, ccRcd);

    unsigned int frameCnt = 0;
    unsigned int frameLimit = cfg.frameLimit;  // number of frames to be processed
    int vchIDFD = -1, vchIDCC = -1;

    chrono::steady_clock::time_point start, end;
    vector<int> infs;
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

    int tVchID = 0;
    while (1) {
        if (sleepPeriodMain > 0)
            this_thread::sleep_for(chrono::milliseconds(sleepPeriodMain));

        if (!isEmptyStreamer(tVchID)) {
            CMat cframe;
            tryPopStreamer(cframe, tVchID);  // get a new cframe

            Mat frame = cframe.frame;  // fix odBatchSize to 1
            int vchID = cframe.vchID;
            frameCnt = cframe.frameCnt;

            if (!checkCmdLogger(cfg, odRcd, fdRcd, ccRcd)) {
                lg("Break loop by checkCmdLogger\n");
                break;
            }

            system_clock::time_point now = system_clock::now();
            int ccNumFrame = 0;

            start = std::chrono::steady_clock::now();

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

                    threadStartFD = frameCnt;  // for the following if-statement
                }

                if (threadStartFD + fdPeriod < frameCnt && fdTh == NULL) {
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

                    ccTh->join();
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

                    threadStartCC = frameCnt;  // for the following if-statement
                }

                if (threadStartCC + ccPeriod < frameCnt && ccTh == NULL) {
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

            end = chrono::steady_clock::now();

            bool needToDraw = needToDrawLogger(vchID);
            if (needToDraw || cfg.recording) {
                if (cfg.odChannels[vchID])
                    drawBoxes(cfg, odRcd, frame, dboxes, vchID);

                if (cfg.fdChannels[vchID])
                    drawFD(cfg, fdRcd, frame, vchID, cfg.fdScoreTh);

                if (cfg.ccChannels[vchID])
                    drawCC(cfg, ccRcd, densityChPre[vchID], frame, vchID);
            }

            if (cfg.recording)
                writeStreamer(frame, vchID);  // write a frame to the output video

            writeDataLogger(cfg, odRcd, fdRcd, ccRcd, frame, frameCnt, vchID, now);

            int inf = chrono::duration_cast<chrono::milliseconds>(end - start).count();

            if (frameCnt % 20 == 0)
                lg(std::format("[{}]Frame{:>4}>  SP({:>2}, {:>3}), Gap(FD: {}, CC: {}),  Buf: {},  Inf: {}ms\n", vchID,
                               frameCnt, sleepPeriodMain, getPeriodStreamer(vchID), fdPeriod, ccPeriod,
                               getUnsafeSizeStreamer(vchID), inf));

            if (frameCnt > 10 && frameCnt < 100)  // skip the start frames and limit the number of elements
                infs.push_back(inf);
        }

        tVchID++;
        if (tVchID >= cfg.numChannels)
            tVchID = 0;

        if (getUnsafeSizeMaxStreamer() > 0) {
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

    destroyModel();     // destroy all models
    destroyLogger();    // destroy logger
    destroyStreamer();  // destroy streamer

    if (infs.size() > 1) {
        float avgInf = accumulate(infs.begin(), infs.end(), 0) / infs.size();
        lg(std::format("\nAverage Inference Time: {}ms\n", avgInf));
    }

    if (cfg.recording) {
        lg("\nOutput file(s):\n");
        for (auto &outFile : cfg.outputFiles)
            lg(std::format("  -{}\n", outFile));
    }

    lg("\nTerminate program!\n");

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

        if (dbox.objID == 8)
            boxColor = Scalar(255, 50, 255);

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

    if (img.rows < 720 || img.cols < 1280)
        return;

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
                    std::format("  CZone {}:{:>7}(L{} {:.2f})", ccZone.ccZoneID, ccZone.ccNums.back(), ccZone.ccLevel,
                                ccZone.avgWindow);
                ccTexts.push_back(text);
            }
        }

        Vis::drawTextBlock(img, Point(img.cols - 555, 500), ccTexts, 1, 2);

        // for demo
        // vector<string> tmp0 = {string("CZone 1")};
        // Vis::drawTextBlock2(img, Point(800, 200), tmp0, 1, 2);
    }
}

void doRunModelFD(Mat &frame, int vchID, uint frameCnt) {
    runModelFD(frame, vchID, frameCnt);
    fdTreadDone = true;
}

void doRunModelCC(Mat &density, Mat &frame, int vchID) {
    // chrono::steady_clock::time_point start, end;
    // start = std::chrono::steady_clock::now();
    runModelCC(density, frame, vchID);
    // end = std::chrono::steady_clock::now();
    // int inf = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    // cout << " Crowd Counting Inference time: " << inf << "ms\n";
    ccTreadDone = true;
}