 //
//  Shape.h
//  MotionTrackingTest
//
//  Created by Kathleen Sullivan on 7/1/15.
//
//

#pragma once
#include "CinderOpenCV.h"

class Shape {
public:
    Shape();
    
    int ID;
    double area;
    cv::Point centroid;
    Boolean matchFound;
    cv::vector<cv::Point> hull;
    int lastFrameSeen;
};
