/*==============================================================================
* Copyright 2022 AIPro Inc.
* Author: Chun-Su Park (cspk@skku.edu)
=============================================================================*/
#pragma once

#ifdef GENERATOR_EXPORTS
#define GENERATOR_API __declspec(dllexport)
#else
#define GENERATOR_API __declspec(dllimport)
#endif

#include <iostream>
#include <opencv2/core.hpp>

#include "global.h"

/** @brief Initialize model
 *
 * @param cfg configuration struct
 * @param odRcd object detection record struct
 * @param fdRcd fire detection record struct
 * @return initialization result(true: success, false: fail)
 */
GENERATOR_API bool initModel(Config &cfg, ODRecord &odRcd, FDRecord &fdRcd);

/** @brief Run detection and PAR models for a frame batch
 *
 * @param dboxesMulP return detected dboxes of all video channels(vchIDs)
 * @param frames batch of frames
 * @param vchIDs vchIDs of batched frames
 * @param frameCnts frameCnts of batched frames
 * @param odScoreTh threshold for filtering low confident detections
 * @param actScoreTh threshold for filtering low confident actions
 * @return flag for the running result(true: success, false: fail)
 */
GENERATOR_API bool runModel(std::vector<std::vector<DetBox>> &dboxesMul, std::vector<cv::Mat> &frames,
                            std::vector<int> &vchIDs, std::vector<uint> &frameCnts, float odScoreTh, float actScoreTh);

/** @brief Run fire detection for a frame batch
 *
 * @param fboxesMulP return detected fboxes of all video channels(vchIDs)
 * @param frames batch of frames
 * @param vchIDs vchIDs of batched frames
 * @param frameCnts frameCnts of batched frames
 * @param fdScoreTh threshold for filtering low confident detections
 * @return flag for the running result(true: success, false: fail)
 */
GENERATOR_API bool runModelFD(std::vector<std::vector<FireBox>> &fboxesMul, std::vector<cv::Mat> &frames,
                              std::vector<int> &vchIDs, std::vector<uint> &frameCnts, float fdScoreTh);

/** @brief Destroy all models
 *
 * @param None
 * @return flag for the destruction result(true: success, false: fail)
 */
GENERATOR_API bool destroyModel();

/** @brief Reset CntLine and Zone configuration
 *
 * @param odRcd record struct
 * @return flag for the reset result(true: success, false: fail)
 */
GENERATOR_API bool resetCntLineAndZone(ODRecord &odRcd);

/** @brief Reset CntLine configuration
 *
 * @param cntLines vector of CntLines
 * @return flag for the reset result(true: success, false: fail)
 */
GENERATOR_API bool resetCntLine(std::vector<CntLine> &cntLines);

/** @brief Reset Zone configuration
 *
 * @param zones vector of Zones
 * @return flag for the reset result(true: success, false: fail)
 */
GENERATOR_API bool resetZone(std::vector<Zone> &zones);

/** @brief Reset CntLine and Zone records
 *
 * @param None
 * @return flag for the reset result(true: success, false: fail)
 */
GENERATOR_API bool resetCntLineAndZoneRecord();

/** @brief Reset CntLine record
 *
 * @param None
 * @return flag for the reset result(true: success, false: fail)
 */
GENERATOR_API bool resetCntLineRecord();

/** @brief Reset Zone record
 *
 * @param None
 * @return flag for the reset result(true: success, false: fail)
 */
GENERATOR_API bool resetZoneRecord();

/** @brief Reset FD
 *
 * @param odRcd record struct
 * @return flag for the reset result(true: success, false: fail)
 */
GENERATOR_API bool resetFD(FDRecord *pFDRcd);

/** @brief Reset FD record
 *
 * @param None
 * @return flag for the reset result(true: success, false: fail)
 */
GENERATOR_API bool resetFDRecord();
