//
//  Shape.cpp
//  MotionTrackingTest
//
//  Created by Kathleen Sullivan on 7/1/15.
//
//

#include "Shape.h"

Shape::Shape() :
centroid( cv::Point() ),
ID(-1),
lastFrameSeen(-1),
matchFound(false)
{
}