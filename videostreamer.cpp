#include "videostreamer.hpp"

#include <string>

#include "opencv2/opencv.hpp"
//#include "util.h"

VideoStreamer::VideoStreamer(Config &cfg, std::vector<CInfo> &cInfo) {
    pCfg = &cfg;
    numChannels = cfg.numChannels;

    inputs = cfg.inputFiles;
    outputs = cfg.outputFiles;

    videoWriters.resize(numChannels);
    captures.resize(numChannels);

    init(cInfo);
}

VideoStreamer::~VideoStreamer() {
}

void VideoStreamer::destroy() {
    for (auto &capture : captures)
        capture.release();

    if (pCfg->recording) {
        for (auto &videoWriter : videoWriters)
            videoWriter.release();
    }
    // std::cout << "destory VideoStreamer ends\n";
}

void VideoStreamer::init(std::vector<CInfo> &cInfo) {
    for (int vchID = 0; vchID < numChannels; vchID++) {
        std::string input = inputs[vchID];

        if (input.empty() || input.length() < 5) {
            cout << std::format("[{}] Wrong address and close this channel: {}\n", vchID, input);
            return;
        }

        cv::VideoCapture &capture = captures[vchID];
        capture.open(input);  // opencv capture 객체 생성

        if (capture.isOpened()) {  //연결 완료
            cout << std::format("[{}] Open: {}\n", vchID, input);

            int frameWidth = capture.get(CAP_PROP_FRAME_WIDTH);
            int frameHeight = capture.get(CAP_PROP_FRAME_HEIGHT);
            float fps = capture.get(CAP_PROP_FPS);

            pCfg->frameHeights[vchID] = frameHeight;
            pCfg->frameWidths[vchID] = frameWidth;
            pCfg->fpss[vchID] = fps;

            if (frameHeight < 0 || frameHeight > 2160 || frameWidth < 0 || frameWidth > 3840) {
                cout << std::format("[{}] Unsupported bitstream: {} {}", vchID, frameHeight, frameWidth);
                exit(-1);
            }

            for (CntLine &c : cInfo[vchID].odRcd.cntLines) {
                if (vchID == c.vchID) {
                    if (c.pts[0].x < 0 || c.pts[0].x >= frameWidth || c.pts[1].x < 0 || c.pts[1].x >= frameWidth) {
                        cout << std::format("[{}] cntLine pt.x error in keepConnected: {} {} {}", vchID, c.pts[0].x,
                                            c.pts[1].x, frameWidth);
                        exit(-1);
                    }
                    if (c.pts[0].y < 0 || c.pts[0].y >= frameHeight || c.pts[1].y < 0 || c.pts[1].y >= frameHeight) {
                        cout << std::format("[{}] cntLine pt.y error in keepConnected: {} {} {}", vchID, c.pts[0].y,
                                            c.pts[1].y, frameHeight);
                        exit(-1);
                    }
                }
            }

            for (Zone &z : cInfo[vchID].odRcd.zones) {
                if (vchID == z.vchID) {
                    for (Point &pt : z.pts) {
                        if (pt.x < 0 || pt.x >= frameWidth) {
                            cout << std::format("[{}] zone pt.x error in keepConnected: {} {}", vchID, pt.x,
                                                frameWidth);
                            exit(-1);
                        }

                        if (pt.y < 0 || pt.y >= frameHeight) {
                            cout << std::format("[{}] zone pt.y error in keepConnected: {} {}", vchID, pt.y,
                                                frameHeight);
                            exit(-1);
                        }
                    }
                }
            }

#ifndef _CPU_INFER
            for (CCZone &z : cInfo[vchID].ccRcd.ccZones) {
                if (vchID == z.vchID) {
                    for (Point &pt : z.pts) {
                        if (pt.x < 0 || pt.x >= frameWidth) {
                            cout << std::format("[{}] ccZone pt.x error in keepConnected: {} {}", vchID, pt.x,
                                                frameWidth);
                            exit(-1);
                        }

                        if (pt.y < 0 || pt.y >= frameHeight) {
                            cout << std::format("[{}] ccZone pt.y error in keepConnected: {} {}", vchID, pt.y,
                                                frameHeight);
                            exit(-1);
                        }
                    }
                }
            }
#endif
            pCfg->odScaleFactors[vchID] =
                std::min((float)pCfg->odNetWidth / frameWidth, (float)pCfg->odNetHeight / frameHeight);
            pCfg->odScaleFactorsInv[vchID] = 1.0f / pCfg->odScaleFactors[vchID];

            pCfg->fdScaleFactors[vchID] =
                std::min((float)pCfg->fdNetWidth / frameWidth, (float)pCfg->fdNetHeight / frameHeight);

            pCfg->ccScaleFactors[vchID] =
                std::min((float)pCfg->ccNetWidth / frameWidth, (float)pCfg->ccNetHeight / frameHeight);

            if (pCfg->recording)
                videoWriters[vchID].open(outputs[vchID], VideoWriter::fourcc('m', 'p', '4', 'v'), fps,
                                         Size(frameWidth, frameHeight));  ///*.mp4 format

            cout << std::format("[{}] ({}, {}), {}\n", vchID, frameWidth, frameHeight, fps);
        } else {
            std::cout << "[" << vchID << "] Can't be opened: " << input << std::endl;
        }
    }
}

bool VideoStreamer::read(Mat &frame, int vchID) {
    cv::VideoCapture &capture = captures[vchID];
    capture.read(frame);

    if (frame.empty())
        return false;

    return true;
}
