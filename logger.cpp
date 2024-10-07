/*==============================================================================
* Copyright 2024 AIPro Inc.
* Author: Chun-Su Park (cspk@skku.edu)
=============================================================================*/

#include "logger.h"

#include "util.h"

Logger::Logger(Config &cfg) : DebugMessage(cfg) {
    logEnable = cfg.logEnable;
    numChannels = cfg.numChannels;
    numPages = ((numChannels - 1) / 9) + 1;  // 9 channels are displayed in one channel
    debugMode = cfg.debugMode;
    targetLiveChannel = -1;

    pageUpdated.resize(numPages, false);
    preTimes.resize(numPages, -1);

    for (int i = 0; i < numPages; i++)
        canvases.push_back(Mat::zeros(1080, 1920, CV_8UC3));

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

    for (int i = 0; i < numChannels; i++)
        preTms.push_back(tm(-1, -1, -1, -1, -1, -1, -1, -1, -1));

    if (!checkDirectories(numChannels)) {
        cout << "Parsing Error!\n";
    }

    createLog();

    if (!logEnable)
        return;

    // for debug
    writeLive = false;
    if (writeLive) {
        lg("<Write live_debug video>\n");
        double fps;

        if (cfg.boostMode)
            fps = 10.0;
        else
            fps = 5.0;

        string filepath = "live_debug.mp4";
        writer.open(filepath, VideoWriter::fourcc('m', 'p', '4', 'v'), fps, Size(1920, 1080));
    }
}

Logger::~Logger() {
    // destroy();
}

void Logger::destroy() {
    // for debug
    if (writeLive)
        writer.release();
}

int Logger::toABTime(string &str) {
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

string Logger::normLogPath(string path) {
#ifdef _WIN32
    return path;
#else  // for linux
    string homeDir = FileUtil::getHomeDirPath();
    return homeDir + "/" + path;
#endif
}

void Logger::newDate(int numChannels, tm *inputTm) {
    tm *curTm;
    if (!inputTm) {
        time_t now = time(NULL);
        curTm = localtime(&now);
    } else
        curTm = inputTm;

    char buf[80];
    strftime(buf, sizeof(buf), "%Y%m%d", curTm);
    string dayInfo = string(buf);

    // lg(std::format("New Date: {}", dayInfo));

    vector<string> targetDirs = {normLogPath(CNT_PATH), normLogPath(FD_PATH), normLogPath(CC_PATH)};
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

            string dirNow = dirChannel + "/now";

            if (exists(dirNow))  // delete an old now directory
                remove_all(dirNow);

            create_directory(dirNow);
        }

        // delete outdated directories
        for (const auto &item : directory_iterator(targetDir)) {
            string dirName = item.path().filename().string();

            if (dirName.length() != 8) {
                lg(std::format("Directory name error. Delete this: {}\n", dirName));
                remove_all(item);
                continue;
            }

            int dirABTime = toABTime(dirName);

            if ((curABTime - dirABTime) > 31) {
                lg(std::format("Remove: {}\n", item.path().string()));
                remove_all(item);
            }
        }
    }

    // delete outdated log files
    for (const auto &item : directory_iterator(normLogPath(LOG_PATH))) {
        string filename = item.path().filename().string();  // with ".txt extension"

        if (filename.length() != 12) {
            lg(std::format("Log file name error. Delete this: {}\n", filename));
            remove(item);
            continue;
        }

        filename.resize(8);

        int dirABTime = toABTime(filename);

        if ((curABTime - dirABTime) > 31) {
            lg(std::format("Remove: {}.txt\n", filename));
            remove(item);
        }
    }
}

bool Logger::checkDirectories(int numChannels) {
    vector<string> dirPaths = {normLogPath(AIPRO_PATH),  normLogPath(ROOT_PATH), normLogPath(CONFIG_PATH),
                               normLogPath(CHIMGS_PATH), normLogPath(CNT_PATH),  normLogPath(CC_PATH),
                               normLogPath(FD_PATH),     normLogPath(LOG_PATH)};

    for (string &dirPath : dirPaths) {
        bool isExist = exists(dirPath);
        if (!isExist)
            create_directory(dirPath);
    }

    string chImgsPath = normLogPath(CHIMGS_PATH);
    string chImgsNowPath = normLogPath(CHIMGS_PATH) + "/now";

    if (exists(chImgsPath))  // delete an old now directory
        remove_all(chImgsPath);

    create_directory(chImgsPath);
    create_directory(chImgsNowPath);

    newDate(numChannels, NULL);

    return true;
}

void Logger::drawCanvase(Mat &frame, int vchID, tm *curTm, int msec) {
    if (vchUpdated[vchID])
        return;

    int page = (vchID / 9);  // ID starts from 0
    Mat &canvase = canvases[page];

    // do not draw
    if (targetLiveChannel != -1 && targetLiveChannel != vchID)
        return;

    if (numChannels == 1 || targetLiveChannel == vchID) {
        resize(frame, canvase, Size(1920, 1080));
    } else {
        Mat resized;
        int tW, tH, elements;

        if (numChannels > 4)
            elements = 3;  // 3x3
        else               // if (numChannels > 1)
            elements = 2;  // 2x2

        tW = 1920 / elements;
        tH = 1080 / elements;

        resize(frame, resized, Size(tW, tH));

        Point tl(vchID % elements * tW, (vchID / elements) % elements * tH);
        resized.copyTo(canvase(Rect(tl, Size(tW, tH))));
    }

    vchUpdated[vchID] = true;
    pageUpdated[page] = true;
}

void Logger::writeIS(Config &cfg, ODRecord &odRcd, CCRecord &ccRcd) {
    string filenameIS = "is.txt";
    string txtPathIS = normLogPath(string(CONFIG_PATH)) + "/" + filenameIS;

    ofstream is(txtPathIS);
    lg(std::format("writeIS: Line = {}, Zone = {}, CZone = {}\n", odRcd.cntLines.size(), odRcd.zones.size(),
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
            is << ccZone.ccLevelThs[0] << " ";
            is << ccZone.ccLevelThs[1] << " ";
            is << ccZone.ccLevelThs[2] << endl;
        }

        // Write scoreThs
        is << 3 << " ";
        is << int(cfg.odScoreTh * 1000) << " ";
        is << int(cfg.fdScoreTh * 1000) << endl;

        is.close();
    }
}

void Logger::writeChInfo() {  // called in CameraStreamer
    string filename = "chinfo.txt";
    string txtPath = normLogPath(string(CONFIG_PATH)) + "/" + filename;

    ofstream chInfo(txtPath);
    lg(std::format("writeChInfo: numChannels = {}\n", pCfg->numChannels));

    if (chInfo.is_open()) {
        for (int c = 0; c < pCfg->numChannels; c++) {
            chInfo << c << " ";
            chInfo << pCfg->frameHeights[c] << " ";
            chInfo << pCfg->frameWidths[c] << " ";
            chInfo << (int)(pCfg->fpss[c] * 100) << endl;
        }

        chInfo.close();
    }
}

void Logger::writeCntLine(ofstream &f, CntLine *c, CntLine *p) {
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

string Logger::getNowCntLine(CntLine *c) {
    string s = std::format("0 {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}", c->clineID, c->vchID, c->isMode,
                           c->totalUL[0][0], c->totalUL[0][1], c->totalUL[0][2], c->totalUL[1][0], c->totalUL[1][1],
                           c->totalUL[1][2], c->totalDR[0][0], c->totalDR[0][1], c->totalDR[0][2], c->totalDR[1][0],
                           c->totalDR[1][1], c->totalDR[1][2]);
    return s;
}

void Logger::writeZone(ofstream &f, Zone *c, Zone *p) {
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

string Logger::getNowZone(Zone *c) {
    string s = std::format("1 {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}", c->zoneID, c->vchID, c->isMode,
                           c->curPeople[0][0], c->curPeople[0][1], c->curPeople[0][2], c->curPeople[1][0],
                           c->curPeople[1][1], c->curPeople[1][2], c->hitMap[0][0], c->hitMap[0][1], c->hitMap[0][2],
                           c->hitMap[1][0], c->hitMap[1][1], c->hitMap[1][2]);

    return s;
}

bool Logger::createLog() {
    static string curLogFilepath;

    // get the current time
    system_clock::time_point now = system_clock::now();
    time_t nowt = system_clock::to_time_t(now);
    tm *curTm = localtime(&nowt);

    char buf[80];
    strftime(buf, sizeof(buf), "%Y%m%d", curTm);
    string dayInfo = string(buf);

    strftime(buf, sizeof(buf), "%T", curTm);
    string timeInfo = string(buf);

    // create a log file
    string newLogFilepath = normLogPath(string(LOG_PATH)) + "/" + dayInfo + ".txt";

    if (curLogFilepath == newLogFilepath)
        return true;  // already created

    if (pCfg->pLogFile) {
        if (pCfg->pLogFile->is_open())
            pCfg->pLogFile->close();

        pCfg->pLogFile = NULL;
    }

    pCfg->pLogFile = new ofstream(newLogFilepath, ios_base::app);
    // pCfg->logFile.open(newLogFilepath, ios_base::app);

    if (!pCfg->pLogFile->is_open()) {
        cout << "Log file error:" << newLogFilepath << endl;
        return false;
    }

    pCfg->lg(std::format("\nStart logging {} at {} ------------------------\n", newLogFilepath, timeInfo));
    curLogFilepath = newLogFilepath;

    return true;
}

bool Logger::needToDraw(int vchID) {
    return !vchUpdated[vchID];
}

bool Logger::checkCmd(Config &cfg, ODRecord &odRcd, FDRecord &fdRcd, CCRecord &ccRcd) {
    if (!logEnable)
        return true;

    string filenameS2E = "cmds2e.txt";
    string txtPathS2E = normLogPath(string(CONFIG_PATH)) + "/" + filenameS2E;
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
            lg(std::format("Get cmd: {}\n", line));

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
                        lg(std::format("cntLine.pts.x error: {} {} {}", cntLine.pts[0].x, cntLine.pts[1].x, fW));
                        return false;
                    }

                    if (cntLine.pts[0].y < 0 || cntLine.pts[0].y >= fH || cntLine.pts[1].y < 0 ||
                        cntLine.pts[1].y >= fH) {
                        lg(std::format("cntLine.pts.y error: {} {} {}", cntLine.pts[0].y, cntLine.pts[1].y, fH));
                        return false;
                    }

                    if (abs(cntLine.pts[0].x - cntLine.pts[1].x) > abs(cntLine.pts[0].y - cntLine.pts[1].y))
                        cntLine.direction = 0;  // horizontal line -> use delta
                                                // y and count U and D
                    else
                        cntLine.direction = 1;  // vertical line -> use delta x
                                                // and count L and R

                    cntLine.init();

                    if (odRcd.vchID != cntLine.vchID) {
                        lg(std::format("vchID errror(insert line): {} {}", odRcd.vchID, cntLine.vchID));
                        return false;
                    }

                    bool duplicated = false;
                    for (auto &item : odRcd.cntLines) {
                        if (item.clineID == cntLine.clineID) {
                            duplicated = true;
                            break;
                        }
                    }

                    if (!duplicated) {
                        odRcd.cntLines.push_back(cntLine);
                    }
                    break;
                }
                case CMD_MODIFY_LINE: {
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
                        lg(std::format("cntLine.pts.x error: {} {} {}", cntLine.pts[0].x, cntLine.pts[1].x, fW));
                        return false;
                    }

                    if (cntLine.pts[0].y < 0 || cntLine.pts[0].y >= fH || cntLine.pts[1].y < 0 ||
                        cntLine.pts[1].y >= fH) {
                        lg(std::format("cntLine.pts.y error: {} {} {}", cntLine.pts[0].y, cntLine.pts[1].y, fH));
                        return false;
                    }

                    if (abs(cntLine.pts[0].x - cntLine.pts[1].x) > abs(cntLine.pts[0].y - cntLine.pts[1].y))
                        cntLine.direction = 0;  // horizontal line -> use delta y and count U and D
                    else
                        cntLine.direction = 1;  // vertical line -> use delta x and count L and R

                    if (odRcd.vchID != cntLine.vchID) {
                        lg(std::format("vchID errror(modify line): {} {}", odRcd.vchID, cntLine.vchID));
                        return false;
                    }

                    for (auto &item : odRcd.cntLines) {
                        if (item.clineID == cntLine.clineID) {
                            item.pts[0] = cntLine.pts[0];
                            item.pts[1] = cntLine.pts[1];
                            item.direction = cntLine.direction;
                            break;
                        }
                    }
                    break;
                }
                case CMD_REMOVE_LINE: {
                    int cLineID, vchID;
                    ss >> cLineID;
                    ss >> vchID;

                    if (vchID >= cfg.numChannels) {
                        lg(std::format("vchID error in remove line: {} {}", vchID, cfg.numChannels));
                        return false;
                    }

                    if (odRcd.vchID != vchID) {
                        lg(std::format("vchID errror(remove line): {} {}", odRcd.vchID, vchID));
                        return false;
                    }

                    for (auto itr = odRcd.cntLines.begin(); itr != odRcd.cntLines.end();) {
                        if (itr->clineID == cLineID)
                            itr = odRcd.cntLines.erase(itr);
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

                    if (zone.vchID != odRcd.vchID) {
                        lg(std::format("vchID errror(insert zone): {} {}", odRcd.vchID, zone.vchID));
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
                            lg(std::format("zone.pts.x error: {} {}", pt.x, fW));
                            return false;
                        }

                        if (pt.y < 0 || pt.y >= fH) {
                            lg(std::format("zone.pts.y error: {} {}", pt.y, fH));
                            return false;
                        }
                    }

                    bool duplicated = false;
                    for (auto &item : odRcd.zones) {
                        if (item.zoneID == zone.zoneID) {
                            duplicated = true;
                            break;
                        }
                    }

                    zone.state = 0;
                    zone.init();

                    if (!duplicated) {
                        odRcd.zones.push_back(zone);
                    }

                    break;
                }
                case CMD_MODIFY_ZONE: {
                    Zone zone;

                    zone.enabled = true;
                    ss >> zone.zoneID;
                    ss >> zone.vchID;
                    ss >> zone.isMode;

                    if (zone.vchID != odRcd.vchID) {
                        lg(std::format("vchID errror(modify zone): {} {}", odRcd.vchID, zone.vchID));
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
                            lg(std::format("zone.pts.x error: {} {}\n", pt.x, fW));
                            return false;
                        }

                        if (pt.y < 0 || pt.y >= fH) {
                            lg(std::format("zone.pts.y error: {} {}\n", pt.y, fH));
                            return false;
                        }
                    }

                    for (auto &item : odRcd.zones) {
                        if (item.zoneID == zone.zoneID && item.isMode == zone.isMode) {
                            std::copy(zone.pts.begin(), zone.pts.end(), item.pts.begin());
                            break;
                        }
                    }
                    break;
                }
                case CMD_REMOVE_ZONE: {
                    int zoneID, vchID;
                    ss >> zoneID;
                    ss >> vchID;

                    if (vchID != odRcd.vchID) {
                        lg(std::format("vchID errror(remove zone): {} {}", odRcd.vchID, vchID));
                        return false;
                    }

                    for (auto itr = odRcd.zones.begin(); itr != odRcd.zones.end();) {
                        if (itr->zoneID == zoneID)
                            itr = odRcd.zones.erase(itr);
                        else
                            ++itr;
                    }
                    break;
                }
                case CMD_INSERT_CCZONE: {
                    CCZone ccZone;

                    ccZone.enabled = true;
                    ccZone.ccLevel = 0;
                    ccZone.preCCLevel = 0;

                    ss >> ccZone.ccZoneID;
                    ss >> ccZone.vchID;

                    if (ccZone.vchID != ccRcd.vchID) {
                        lg(std::format("vchID errror(insert ccZone): {} {}", ccRcd.vchID, ccZone.vchID));
                        return false;
                    }
#ifdef _CPU_INFER
                    // Store only one ccZone for each vchID in CPU mode
                    if (!ccRcd.ccZones.size() > 0) {
                        cout << "CCZone error!: Only one ccZone can be used in cpu mode\n";
                        continue;
                    }
#endif
                    int fH = cfg.frameHeights[ccZone.vchID];
                    int fW = cfg.frameWidths[ccZone.vchID];

                    for (int i = 0; i < 4; i++) {
                        Point pt;
                        ss >> pt.x;
                        ss >> pt.y;
                        ccZone.pts.push_back(pt);

                        if (pt.x < 0 || pt.x >= fW) {
                            lg(std::format("zone.pts.x error: {} {}\n", pt.x, fW));
                            return false;
                        }

                        if (pt.y < 0 || pt.y >= fH) {
                            lg(std::format("zone.pts.y error: {} {}\n", pt.y, fH));
                            return false;
                        }
                    }

                    ss >> ccZone.ccLevelThs[0];
                    ss >> ccZone.ccLevelThs[1];
                    ss >> ccZone.ccLevelThs[2];

                    if (ccZone.ccLevelThs[0] > ccZone.ccLevelThs[1] || ccZone.ccLevelThs[1] > ccZone.ccLevelThs[2]) {
                        cout << "ccLevelThs should be ordered.\n";
                        return false;
                    }

                    ccZone.ccNums.resize(cfg.ccWindowSize, 0);

                    ccZone.maxCC = 0;
                    ccZone.maxCCDay = 0;

                    ccZone.init();

                    // init mask with empty Mat. This is generated in
                    // CrowdCounter::runModel.
                    ccZone.mask = cv::Mat();
#ifdef _CPU_INFER
                    ccZone.canvas = cv::Mat();
                    ccZone.roiCanvas = cv::Mat();
#endif

                    bool duplicated = false;
                    for (auto &item : ccRcd.ccZones) {
                        if (item.ccZoneID == ccZone.ccZoneID) {
                            duplicated = true;
                            break;
                        }
                    }

                    if (!duplicated)
                        ccRcd.ccZones.push_back(ccZone);

                    break;
                }
                case CMD_MODIFY_CCZONE: {
                    CCZone ccZone;

                    ccZone.enabled = true;
                    ccZone.ccLevel = 0;
                    ccZone.preCCLevel = 0;

                    ss >> ccZone.ccZoneID;
                    ss >> ccZone.vchID;

                    if (ccZone.vchID != ccRcd.vchID) {
                        lg(std::format("vchID errror(modify ccZone): {} {}", ccRcd.vchID, ccZone.vchID));
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
                            lg(std::format("zone.pts.x error: {} {}\n", pt.x, fW));
                            return false;
                        }

                        if (pt.y < 0 || pt.y >= fH) {
                            lg(std::format("zone.pts.y error: {} {}\n", pt.y, fH));
                            return false;
                        }
                    }

                    ss >> ccZone.ccLevelThs[0];
                    ss >> ccZone.ccLevelThs[1];
                    ss >> ccZone.ccLevelThs[2];

                    for (auto &item : ccRcd.ccZones) {
                        if (item.ccZoneID == ccZone.ccZoneID) {
                            std::copy(ccZone.pts.begin(), ccZone.pts.end(), item.pts.begin());
                            item.ccLevelThs[0] = ccZone.ccLevelThs[0];
                            item.ccLevelThs[1] = ccZone.ccLevelThs[1];
                            item.ccLevelThs[2] = ccZone.ccLevelThs[2];

                            item.mask = cv::Mat();
#ifdef _CPU_INFER
                            item.canvas = cv::Mat();
                            item.roiCanvas = cv::Mat();
#endif
                            break;
                        }
                    }
                    break;
                }
                case CMD_REMOVE_CCZONE: {
                    int ccZoneID, vchID;

                    ss >> ccZoneID;
                    ss >> vchID;

                    if (vchID != ccRcd.vchID) {
                        lg(std::format("vchID errror(remove ccZone): {} {}", ccRcd.vchID, vchID));
                        return false;
                    }

                    for (auto itr = ccRcd.ccZones.begin(); itr != ccRcd.ccZones.end();) {
                        if (itr->ccZoneID == ccZoneID) {
                            itr->enabled = false;
                            itr->vchID = -1;
                            itr = ccRcd.ccZones.erase(itr);
                        } else
                            ++itr;
                    }
                    break;
                }
                case CMD_UPDATE_SCORETHS: {
                    int odScoreTh, fdScoreTh;
                    ss >> odScoreTh;
                    ss >> fdScoreTh;

                    if (odScoreTh >= 1000 || fdScoreTh >= 1000) {
                        lg(std::format("ScoreThs error: {} {}", odScoreTh, fdScoreTh));
                        return false;
                    }

                    cfg.odScoreTh = (odScoreTh / 1000.0f);
                    cfg.fdScoreTh = (fdScoreTh / 1000.0f);

                    break;
                }
                case CMD_ENABLE_TARGET_LIVE_CHANNEL: {
                    ss >> targetLiveChannel;

                    if (targetLiveChannel < 0 || targetLiveChannel >= numChannels) {
                        lg(std::format("targetLiveChannel error: {} {}", targetLiveChannel, numChannels));
                        return false;
                    }

                    for (int i = 0; i < numChannels; i++) {
                        if (i == targetLiveChannel)
                            vchUpdated[i] = false;  // need to draw
                        else
                            vchUpdated[i] = true;  // don't need to draw
                    }

                    break;
                }
                case CMD_DISABLE_TARGET_LIVE_CHANNEL: {
                    targetLiveChannel = -1;

                    for (int i = 0; i < numChannels; i++)
                        vchUpdated[i] = false;  // need to draw

                    // init canvases
                    for (int p = 0; p < numPages; p++)
                        canvases[p].setTo(cv::Scalar(0, 0, 0));

                    break;
                }
                case CMD_CLEARLOG: {
                    remove_all(normLogPath(CNT_PATH));
                    remove_all(normLogPath(FD_PATH));
                    remove_all(normLogPath(CC_PATH));

                    checkDirectories(numChannels);

                    for (auto &zone : odRcd.zones)
                        zone.init();

                    for (auto &cntLine : odRcd.cntLines)
                        cntLine.init();

                    fdRcd.fireProbs.clear();
                    fdRcd.fireProbs.clear();
                    fdRcd.fireProbs.resize(cfg.fdWindowSize, 0.0f);
                    fdRcd.smokeProbs.resize(cfg.fdWindowSize, 0.0f);
                    fdRcd.fireEvent = 0;
                    fdRcd.smokeEvent = 0;
                    fdRcd.afterFireEvent = 0;

                    ccRcd.ccNumFrames.clear();
                    ccRcd.ccNumFrames.resize(cfg.ccWindowSize);

                    for (CCZone &ccZone : ccRcd.ccZones) {
                        ccZone.ccNums.clear();
                        ccZone.ccNums.resize(cfg.ccWindowSize, 0);

                        ccZone.maxCC = 0;
                        ccZone.maxCCDay = 0;

                        ccZone.init();
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

void Logger::writeData(Config &cfg, ODRecord &odRcd, FDRecord &fdRcd, CCRecord &ccRcd, Mat &frame,
                       unsigned int &frameCnt, int vchID, system_clock::time_point now) {
    if (!logEnable)
        return;

    // write cmde2s.txt to update vch states
    if (vchStatesPre != cfg.vchStates) {
        vchStatesPre = cfg.vchStates;

        string filename = "cmde2s.txt";
        string txtPathCmde2s = normLogPath(string(CONFIG_PATH)) + "/" + filename;
        ofstream logFileCmde2s(txtPathCmde2s);

        if (logFileCmde2s.is_open()) {
            logFileCmde2s << "0 " << numChannels;
            lg("0 ");

            for (int &state : vchStatesPre) {
                logFileCmde2s << " " << state;
                lg(std::format(" {}", state));
            }

            lg(" <= Write cmde2s.txt\n");
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
        string imgPath = normLogPath(string(CHIMGS_PATH)) + "/" + filename;
        imwrite(imgPath, frame, {cv::ImwriteFlags::IMWRITE_JPEG_QUALITY, 80});

        writeChImgs[vchID] = false;
    }

    drawCanvase(frame, vchID, curTm, msec);

    // generate a 30 min log and a one day log
    // if ((preTm.tm_min == 29 && curTm->tm_min == 30) || (preTm.tm_min != -1 &&
    // preTm.tm_min != curTm->tm_min)){
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

        txtPathCnt = normLogPath(string(CNT_PATH)) + "/" + dayInfoSelected + "/" + to_string(vchID) + "/" + filename;
        txtPathFD = normLogPath(string(FD_PATH)) + "/" + dayInfoSelected + "/" + to_string(vchID) + "/" + filename;
        txtPathCC = normLogPath(string(CC_PATH)) + "/" + dayInfoSelected + "/" + to_string(vchID) + "/" + filename;

        if (cfg.odChannels[vchID]) {
            ofstream logFileCnt(txtPathCnt);

            if (logFileCnt.is_open()) {
                for (int n = 0; n < odRcd.cntLines.size(); n++) {
                    if (odRcd.cntLines[n].vchID != vchID)
                        continue;

                    auto &c = odRcd.cntLines[n];
                    writeCntLine(logFileCnt, &c);
                }

                for (int n = 0; n < odRcd.zones.size(); n++) {
                    if (odRcd.zones[n].vchID != vchID)
                        continue;

                    auto &c = odRcd.zones[n];
                    writeZone(logFileCnt, &c);
                }
            }

            logFileCnt.close();
        }

        if (cfg.fdChannels[vchID]) {
            ofstream logFileFD(txtPathFD);

            if (logFileFD.is_open()) {
                string text = std::format("{} {} {} {}\n", accNumFire, accNumSmoke, maxFireProb, maxSmokeProb);
                logFileFD << text;
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

                    string text = std::format("{} {} {} {} {}\n", ccZone.ccZoneID, ccZone.maxCC, ccZone.accCCLevels[0],
                                              ccZone.accCCLevels[1], ccZone.accCCLevels[2]);
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
            dayChanged = true;  // checked for all channels, but log is created
            // just once(see createLog())

            newDate(cfg.numChannels, curTm);  // prepare a new directory for curTm

            string filename = "day.txt";
            string txtPathCnt =
                normLogPath(string(CNT_PATH)) + "/" + dayInfoPre + "/" + to_string(vchID) + "/" + filename;
            string txtPathFD =
                normLogPath(string(FD_PATH)) + "/" + dayInfoPre + "/" + to_string(vchID) + "/" + filename;
            string txtPathCC =
                normLogPath(string(CC_PATH)) + "/" + dayInfoPre + "/" + to_string(vchID) + "/" + filename;

            if (cfg.odChannels[vchID]) {
                ofstream logFileCnt(txtPathCnt);

                if (logFileCnt.is_open()) {
                    for (int n = 0; n < odRcd.cntLines.size(); n++) {
                        if (odRcd.cntLines[n].vchID != vchID)
                            continue;

                        auto &c = odRcd.cntLines[n];
                        writeCntLine(logFileCnt, &c);

                        c.init();
                    }

                    for (int n = 0; n < odRcd.zones.size(); n++) {
                        if (odRcd.zones[n].vchID != vchID)
                            continue;

                        auto &c = odRcd.zones[n];
                        writeZone(logFileCnt, &c);

                        c.init();
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

                        string text =
                            std::format("{} {} {} {} {}\n", ccZone.ccZoneID, ccZone.maxCCDay, ccZone.accCCLevelsDay[0],
                                        ccZone.accCCLevelsDay[1], ccZone.accCCLevelsDay[2]);

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
        deleteNowLive(curTm->tm_sec, cfg.boostMode);  // delete old file

        if (cfg.odChannels[vchID]) {
            writeNowOD(odRcd, vchID, dayInfo, curTm->tm_sec);

            bool saveImg = false;
            string filenameRet = std::format("{:02}{:02}{:02}.txt", curTm->tm_hour, curTm->tm_min, curTm->tm_sec);
            string txtPathRet =
                normLogPath(string(CNT_PATH)) + "/" + dayInfo + "/" + to_string(vchID) + "/txts/" + filenameRet;

            for (int n = 0; n < odRcd.cntLines.size(); n++) {
                CntLine &c = odRcd.cntLines[n];

                if (c.vchID != vchID || c.isMode != IS_RESTRICTED_AREA)
                    continue;

                // only for restricted-mode lines with vchID
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

            for (int n = 0; n < odRcd.zones.size(); n++) {
                Zone &z = odRcd.zones[n];

                if (odRcd.zones[n].vchID != vchID || z.isMode != IS_RESTRICTED_AREA)
                    continue;

                // only for restricted-mode zones with vchID
                int curTotal = z.getTotal();

                if (z.preTotal > 0 && curTotal > 0) {
                    if (z.state == 0) {  // check 0, 1, 1
                        ofstream logFileRet(txtPathRet, ios_base::app);

                        if (logFileRet.is_open()) {
                            writeZone(logFileRet, &z);
                            logFileRet.close();

                            saveImg = true;
                        }
                    }
                    z.state = 2;
                } else if (z.preTotal == 0 && curTotal == 0 && z.state > 0)  // check 0, 0, 0
                    z.state--;

                z.preTotal = curTotal;
            }

            if (saveImg) {
                string filenameRetImg =
                    std::format("{:02}{:02}{:02}.jpg", curTm->tm_hour, curTm->tm_min, curTm->tm_sec);
                string imgPathRet =
                    normLogPath(string(CNT_PATH)) + "/" + dayInfo + "/" + to_string(vchID) + "/imgs/" + filenameRetImg;

                imwrite(imgPathRet, frame, {cv::ImwriteFlags::IMWRITE_JPEG_QUALITY, 70});
            }
        }

        if (cfg.fdChannels[vchID]) {
            int fireEvent = fdRcd.fireEvent;
            int smokeEvent = fdRcd.smokeEvent;

            int fireProb = fdRcd.fireProbs.back() * 1000;
            int smokeProb = fdRcd.smokeProbs.back() * 1000;
            int &afterFireEvent = fdRcd.afterFireEvent;

            if ((fireProb > cfg.fdScoreTh) || (smokeProb > cfg.fdScoreTh)) {
                string filename = std::format("{:02}{:02}{:02}.txt", curTm->tm_hour, curTm->tm_min, curTm->tm_sec);
                string txtPathFD =
                    normLogPath(string(FD_PATH)) + "/" + dayInfo + "/" + to_string(vchID) + "/txts/" + filename;
                ofstream logFileFD(txtPathFD);
                int isFirst = (afterFireEvent == 0) ? 1 : 0;

                if (logFileFD.is_open()) {
                    string text = std::format("{} {} {} {} {}\n", isFirst, fireEvent, smokeEvent, fireProb, smokeProb);
                    logFileFD << text;

                    logFileFD.close();
                }

                // if (fireEvent || smokeEvent) {
                if (isFirst || curTm->tm_sec % 10 == 0) {
                    // if (curTm->tm_sec % 10 == 0) {
                    string filenameImg =
                        std::format("{:02}{:02}{:02}.jpg", curTm->tm_hour, curTm->tm_min, curTm->tm_sec);
                    string imgPathFD =
                        normLogPath(string(FD_PATH)) + "/" + dayInfo + "/" + to_string(vchID) + "/imgs/" + filenameImg;

                    imwrite(imgPathFD, frame, {cv::ImwriteFlags::IMWRITE_JPEG_QUALITY, 70});
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

            accNumFires[vchID] += fireEvent;
            accNumSmokes[vchID] += smokeEvent;

            // should be here to include the current fire and smoke events
            writeNowFD(vchID, dayInfo, curTm->tm_sec);
        }

        if (cfg.ccChannels[vchID]) {
            writeNowCC(ccRcd, vchID, dayInfo, curTm->tm_sec);

            vector<string> ccEventLogs;
            for (CCZone &ccZone : ccRcd.ccZones) {
                if (ccZone.vchID != vchID)
                    continue;

                int curCCNum = ccZone.ccNums.back();

                // update parameters for 30 min file
                if (curCCNum > ccZone.maxCC)
                    ccZone.maxCC = curCCNum;

                if (ccZone.preCCLevel < ccZone.ccLevel) {  // when ccLevel gets higher
                    ccZone.accCCLevels[ccZone.ccLevel - 1]++;

                    string text =
                        std::format("{} {} {} {}\n", ccZone.ccZoneID, curCCNum, ccZone.ccLevel, ccZone.preCCLevel);
                    ccEventLogs.push_back(text);
                }

                ccZone.preCCLevel = ccZone.ccLevel;
            }

            if (ccEventLogs.size() > 0) {
                string filenameCCEvent =
                    std::format("{:02}{:02}{:02}.txt", curTm->tm_hour, curTm->tm_min, curTm->tm_sec);
                string txtPathCCEvent =
                    normLogPath(string(CC_PATH)) + "/" + dayInfo + "/" + to_string(vchID) + "/txts/" + filenameCCEvent;
                ofstream logFileCCEvent(txtPathCCEvent);

                if (logFileCCEvent.is_open()) {
                    for (string &ccEventLog : ccEventLogs)
                        logFileCCEvent << ccEventLog;

                    logFileCCEvent.close();
                }

                string filenameCCEventImg =
                    std::format("{:02}{:02}{:02}.jpg", curTm->tm_hour, curTm->tm_min, curTm->tm_sec);
                string imgPathCCEventImg = normLogPath(string(CC_PATH)) + "/" + dayInfo + "/" + to_string(vchID) +
                                           "/imgs/" + filenameCCEventImg;
                imwrite(imgPathCCEventImg, frame, {cv::ImwriteFlags::IMWRITE_JPEG_QUALITY, 70});
            }
        }
    }

    *preTm = *curTm;

    writeNowLive(curTm->tm_sec, msec, cfg.boostMode);

    if (dayChanged)
        createLog();
}

void Logger::writeNowOD(ODRecord &odRcd, int vchID, string dayInfo, int sec) {
    string filenameODNow = std::format("{}.txt", sec % 10);
    string txtPathODNow =
        normLogPath(string(CNT_PATH)) + "/" + dayInfo + "/" + to_string(vchID) + "/now/" + filenameODNow;
    ofstream logFileODNow(txtPathODNow);

    if (logFileODNow.is_open()) {
        for (int n = 0; n < odRcd.cntLines.size(); n++) {
            if (odRcd.cntLines[n].vchID != vchID)
                continue;

            CntLine &c = odRcd.cntLines[n];
            writeCntLine(logFileODNow, &c);
        }

        for (int n = 0; n < odRcd.zones.size(); n++) {
            if (odRcd.zones[n].vchID != vchID)
                continue;

            Zone &z = odRcd.zones[n];
            writeZone(logFileODNow, &z);
        }

        logFileODNow.close();
    }
}

// void Logger::getNowOD(vector<string>& nowOD) {
//    ODRecord* pOdRcd = pCfg->pOdRcd;
//
//    for (int n = 0; n < pOdRcd->cntLines.size(); n++) {
//        CntLine& c = pOdRcd->cntLines[n];
//        string text = getNowCntLine(&c);
//        nowOD.push_back(text);
//    }
//
//    for (int n = 0; n < pOdRcd->zones.size(); n++) {
//        Zone& z = pOdRcd->zones[n];
//        string text = getNowZone(&z);
//        nowOD.push_back(text);
//    }
//}

void Logger::writeNowFD(int vchID, string dayInfo, int sec) {
    string filenameFDNow = std::format("{}.txt", sec % 10);
    string txtPathFDNow =
        normLogPath(string(FD_PATH)) + "/" + dayInfo + "/" + to_string(vchID) + "/now/" + filenameFDNow;
    ofstream logFileFDNow(txtPathFDNow);

    if (logFileFDNow.is_open()) {
        string text = std::format("{} {} {} {}\n", accNumFires[vchID], accNumSmokes[vchID], maxFireProbs[vchID],
                                  maxSmokeProbs[vchID]);
        logFileFDNow << text;

        logFileFDNow.close();
    }
}

// void Logger::getNowFD(vector<string>& nowFD) {
//    for (int vchID = 0; vchID < numChannels; vchID++) {
//        string text = std::format("{} {} {} {} {}\n", vchID, accNumFires[vchID], accNumSmokes[vchID],
//        maxFireProbs[vchID],
//            maxSmokeProbs[vchID]);
//        nowFD.push_back(text);
//    }
//}

void Logger::writeNowCC(CCRecord &ccRcd, int vchID, string dayInfo, int sec) {
    string filenameCCNow = std::format("{}.txt", sec % 10);
    string txtPathCCNow =
        normLogPath(string(CC_PATH)) + "/" + dayInfo + "/" + to_string(vchID) + "/now/" + filenameCCNow;
    ofstream logFileCCNow(txtPathCCNow);

    if (logFileCCNow.is_open()) {
        for (CCZone &ccZone : ccRcd.ccZones) {
            if (ccZone.vchID != vchID)
                continue;

            int curCCNum = ccZone.ccNums.back();

            string text = std::format("{} {} {} {} {} {} {}\n", ccZone.ccZoneID, curCCNum, ccZone.maxCC, ccZone.ccLevel,
                                      ccZone.accCCLevels[0], ccZone.accCCLevels[1], ccZone.accCCLevels[2]);

            logFileCCNow << text;
        }

        logFileCCNow.close();
    }
}

// void Logger::getNowCC(vector<string>& nowCC) {
//    CCRecord* pCcRcd = pCfg->pCcRcd;
//
//    for (CCZone& ccZone : pCcRcd->ccZones) {
//        int curCCNum = ccZone.ccNums.back();
//
//        string text = std::format("{} {} {} {} {} {} {} {}\n", ccZone.ccZoneID, ccZone.vchID, curCCNum, ccZone.maxCC,
//        ccZone.ccLevel, ccZone.accCCLevels[0],
//            ccZone.accCCLevels[1], ccZone.accCCLevels[2]);
//
//        nowCC.push_back(text);
//    }
//}

void Logger::writeNowLive(int sec, int msec, bool boostMode) {
    for (int p = 0; p < numPages; p++) {
        if (pageUpdated[p]) {
            int msecD;

            if (boostMode)
                msecD = msec / 100;  // save at 10 fps
            else
                msecD = msec / 200;  // save at 5 fps

            if (preTimes[p] != msecD) {
                preTimes[p] = msecD;

                string filename = std::format("{}p{:02d}{}.jpg", p, sec % 60, msecD);
                // string filename = std::format("{}p{}{}.jpg", p, sec % 10, msecD);
                string imgPath = normLogPath(string(CHIMGS_PATH)) + "/now/" + filename;

                // for debug
                if (debugMode) {
                    rectangle(canvases[p], Rect(0, 0, 200, 28), Scalar(255, 255, 255), -1);
                    putText(canvases[p], filename, Point(0, 24), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 0));
                }

                if (writeLive && p == 0)
                    writer << canvases[p];

                imwrite(imgPath, canvases[p], {cv::ImwriteFlags::IMWRITE_JPEG_QUALITY, 80});

                pageUpdated[p] = false;
                std::fill(vchUpdated.begin(), vchUpdated.end(), false);
            }
        }
    }
}

// void Logger::getNowLive(vector<Mat>& nowMats) {
//    nowMats.resize(numPages);
//
//    for (int p = 0; p < numPages; p++) {
//        canvases[p].copyTo(nowMats[p]);
//        pageUpdated[p] = false;
//    }
//
//    std::fill(vchUpdated.begin(), vchUpdated.end(), false);
//}

void Logger::deleteNowLive(int sec, bool boostMode) {
    int msec[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

    for (int p = 0; p < numPages; p++) {
        int lastIdx = boostMode ? 10 : 5;

        for (int i = 0; i < lastIdx; i++) {
            int firstDigit = ((sec + 60) - 6) % 60;
            int sencondDigit = i;

            string filename = std::format("{}p{:02d}{}.jpg", p, firstDigit, sencondDigit);
            string imgPath = normLogPath(string(CHIMGS_PATH)) + "/now/" + filename;

            if (exists(imgPath))  // delete an old now directory
                remove(imgPath);
        }
    }
}