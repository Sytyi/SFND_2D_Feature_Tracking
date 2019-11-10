/* INCLUDES FOR THIS PROJECT */
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <deque>
#include <cmath>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"
#include "matching2D.hpp"

using namespace std;

/* MAIN PROGRAM */
int main(int argc, const char *argv[])
{

    /* INIT VARIABLES AND DATA STRUCTURES */

    // data location
    string dataPath = "../";

    // camera
    string imgBasePath = dataPath + "images/";
    string imgPrefix = "KITTI/2011_09_26/image_00/data/000000"; // left camera, color
    string imgFileType = ".png";
    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 9;   // last file index to load
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

    // misc
    int dataBufferSize = 2;       // no. of images which are held in memory (ring buffer) at the same time
    std::deque<DataFrame> dataBuffer; // list of data frames which are held in memory at the same time
    bool bVis = false;            // visualize results

    /* MAIN LOOP OVER ALL IMAGES */
    vector<string> detectorTypes = {"SHITOMASI", "HARRIS", "FAST", "BRISK", "ORB", "AKAZE", "SIFT" };
    vector<string> descriptorTypes = {"BRISK", "BRIEF", "ORB", "FREAK", "AKAZE", "SIFT"};

    string matcherType = "MAT_FLANN";        // MAT_BF, MAT_FLANN
    string descriptorMatchType = "DES_BINARY"; // DES_BINARY, DES_HOG
    string selectorType = "SEL_KNN";       // SEL_NN, SEL_KNN

    // I've added a loop to get logs for every combination

    for( auto detectorType: detectorTypes )
        for( auto descriptorType: descriptorTypes ) {
            cout << "==================================" << endl;
            cout << "Detector: " << detectorType << "; Decriptor: " << descriptorType << endl;
            if (descriptorType == descriptorTypes[4] && detectorType != detectorTypes[5] )
            {
                cout << descriptorType << " not compatible with " << detectorType << endl;
                continue;
            }
            if (descriptorType == descriptorTypes[2] && detectorType == detectorTypes[6] )
            {
                cout << descriptorType << " not compatible with " << detectorType << endl;
                continue;
            }

            dataBuffer.clear();
            vector<pair<int,int>> keypointsSize;
            vector<double> detectorTimes;
            vector<double> descriptorTimes;
            vector<pair<int,double>> matchTimes;

            for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex++) {
                /* LOAD IMAGE INTO BUFFER */

                // assemble filenames for current index
                ostringstream imgNumber;
                imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
                string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

                // load image from file and convert to grayscale
                cv::Mat img, imgGray;
                img = cv::imread(imgFullFilename);
                cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

                // push image into data frame buffer
                DataFrame frame;
                frame.cameraImg = imgGray;
                dataBuffer.push_back(frame);
                if (dataBuffer.size() > dataBufferSize) {
                    dataBuffer.pop_front();
                }
                cout << "#1 : LOAD IMAGE " << imgNumber.str() << " INTO BUFFER " << dataBuffer.size() << " done"
                     << endl;

                /* DETECT IMAGE KEYPOINTS */
                // extract 2D keypoints from current image
                vector<cv::KeyPoint> keypoints; // create empty feature list for current image
                double tDet;
                if (detectorType.compare("SHITOMASI") == 0) {
                    tDet = detKeypointsShiTomasi(keypoints, imgGray, bVis);
                } else if (detectorType.compare("HARRIS") == 0) {
                    tDet = detKeypointsHarris(keypoints, imgGray, bVis);
                } else {
                    tDet = detKeypointsModern(keypoints, imgGray, detectorType, bVis);
                }
                detectorTimes.push_back(tDet);
                // only keep keypoints on the preceding vehicle
                bool bFocusOnVehicle = true;
                cv::Rect vehicleRect(535, 180, 180, 150);
                int beforeFilteringSize = keypoints.size();
                if (bFocusOnVehicle) {
                    auto it = keypoints.begin();
                    while (it != keypoints.end()) {
                        if (!vehicleRect.contains(it->pt)) {
                            it = keypoints.erase(it);
                        } else
                            ++it;
                    }
                }
                keypointsSize.push_back(pair<int,int>(beforeFilteringSize,keypoints.size()));

                // optional : limit number of keypoints (helpful for debugging and learning)
                bool bLimitKpts = false;
                if (bLimitKpts) {
                    int maxKeypoints = 50;

                    if (detectorType.compare("SHITOMASI") ==
                        0) { // there is no response info, so keep the first 50 as they are sorted in descending quality order
                        keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
                    }
                    cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
                    cout << " NOTE: Keypoints have been limited!" << endl;
                }

                // push keypoints and descriptor for current frame to end of data buffer
                (dataBuffer.end() - 1)->keypoints = keypoints;
                cout << "#2 : DETECT " << keypoints.size() << " KEYPOINTS; before filtering = " << beforeFilteringSize
                     << " done" << endl;

                /* EXTRACT KEYPOINT DESCRIPTORS */

                cv::Mat descriptors;
                double tDesc = descKeypoints((dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->cameraImg, descriptors,
                              descriptorType);
                descriptorTimes.push_back(tDesc);
                // push descriptors for current frame to end of data buffer
                (dataBuffer.end() - 1)->descriptors = descriptors;

                cout << "#3 : EXTRACT DESCRIPTORS done" << endl;

                if (dataBuffer.size() > 1) // wait until at least two images have been processed
                {

                    /* MATCH KEYPOINT DESCRIPTORS */

                    vector<cv::DMatch> matches;
                    double tMatch = matchDescriptors((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
                                     (dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
                                     matches, descriptorMatchType, matcherType, selectorType);
                    matchTimes.push_back(pair<int,double>(matches.size(),tMatch));
                    // store matches in current data frame
                    (dataBuffer.end() - 1)->kptMatches = matches;

                    cout << "#4 : MATCH KEYPOINT DESCRIPTORS done" << endl;

                    // visualize matches between current and previous image
                    bVis = true;
                    if (bVis) {
                        cv::Mat matchImg = ((dataBuffer.end() - 1)->cameraImg).clone();
                        cv::drawMatches((dataBuffer.end() - 2)->cameraImg, (dataBuffer.end() - 2)->keypoints,
                                        (dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->keypoints,
                                        matches, matchImg,
                                        cv::Scalar::all(-1), cv::Scalar::all(-1),
                                        vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

                        string windowName = "Matching keypoints between two camera images: "
                                + detectorType + " : " + descriptorType;
                        cv::namedWindow(windowName, 7);
                        cv::imshow(windowName, matchImg);
                        cout << "Press key to continue to next image" << endl;
                        cv::waitKey(0); // wait for key to be pressed
                    }
                    bVis = false;
                }else
                {
                    matchTimes.push_back(pair<int,double>(0,0));
                }

            } // eof loop over all images

            // the part I'm not proud of.
            // printing csv in the log, so I will parse it and add to report
            cout << "--------------------" << endl;
            cout << "Detector: " << detectorType << "; Decriptor: " << descriptorType << endl;
            cout << "Image#; CarKP; Frame KP; tDet; tDesc; nMatches; tMatch;" << endl;

            for( int i = 0; i < keypointsSize.size(); ++i)
            {
                auto kpV = keypointsSize[i];
                auto matchV = matchTimes[i];
                cout << i << ";" << std::fixed
                << setfill('0') << setw(4) << kpV.second << ";"
                << setfill('0') << setw(4) << kpV.first << ";"
                << setfill('0') << setw(6) << setprecision(3) << detectorTimes[i] << ";"
                << descriptorTimes[i] <<";"
                << setfill('0') << setw(4) << matchV.first << ";"
                << setfill('0') << setw(6) << setprecision(3) << matchV.second << endl;
            }

        }
    return 0;
}
