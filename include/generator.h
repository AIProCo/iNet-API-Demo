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
 * @return initialization result(true: success, false: fail)
 */
GENERATOR_API bool initModel(Config& cfg);

/** @brief Run detection and PAR models for a frame batch
 *
 * @param dboxesMul return detected dboxes of all video channels(vchIDs)
 * @param frames batch of frames
 * @param vchIDs vchIDs of batched frames
 * @param frameCnts frameCnts of batched frames
 * @param odScoreTh threshold for filtering low confident detections
 * @param framesStory the number of False-Negative detections, during which track_id will be kept
 * @param maxDist max distance in pixels between previous and current detection, to keep the same track_id
 * @return flag for the running result(true: success, false: fail)
 */
GENERATOR_API bool runModel(std::vector<std::vector<DetBox>>& dboxesMul, std::vector<cv::Mat>& frames,
    std::vector<int>& vchIDs, std::vector<uint>& frameCnts, float odScoreTh,
    int framesStory, int maxDist);

/** @brief Run Pose and Action models for the detected dboxesMul
 *
 * @param dboxesMul return extracted Skeletons for the inserted dboxesMul
 * @param frames batch of frames
 * @param vchIDs vchIDs of batched frames
 * @param frameCnts frameCnts of batched frames
 * @param dboxesMul detected dboxes of batched frames in runModel
 * @param actScoreTh threshold for filtering low confident actions
 * @return flag for the running result(true: success, false: fail)
 */
GENERATOR_API bool runModelAct(std::vector<std::vector<DetBox>>& dboxesMul, std::vector<cv::Mat>& frames,
    std::vector<int>& vchIDs, std::vector<uint>& frameCnts, float actScoreTh);

/** @brief Destroy all models
 *
 * @param None
 * @return flag for the destruction result(true: success, false: fail)
 */
GENERATOR_API bool destroyModel();

/** @brief Reset CntLine and Zone configuration
 *
 * @param None
 * @return flag for the reset result(true: success, false: fail)
 */
GENERATOR_API bool resetCntLineAndZone(Config& cfg);

/** @brief Reset records
 *
 * @param None
 * @return flag for the reset result(true: success, false: fail)
 */
GENERATOR_API bool resetRecord();
