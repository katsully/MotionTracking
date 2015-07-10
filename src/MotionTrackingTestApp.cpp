#include "cinder/app/AppNative.h"
#include "cinder/gl/Texture.h"
#include "cinder/gl/gl.h"
#include "Cinder-OpenNI.h"
#include "CinderOpenCV.h"
#include "cinder/params/Params.h"
#include "Shape.h"

using namespace ci;
using namespace ci::app;
using namespace std;

class MotionTrackingTestApp : public AppNative {
  public:
	void setup();
    void prepareSettings( Settings* settings );
   // void keyDown( KeyEvent event );
    void onDepth( openni::VideoFrameRef frame, const OpenNI::DeviceOptions& deviceOptions );
    void onColor( openni::VideoFrameRef frame, const OpenNI::DeviceOptions& deviceOptions );
    vector< Shape > getEvaluationSet( vector< vector<cv::Point> > rawContours, int minimalArea, int maxArea );
    Shape* findNearestMatch( Shape trackedShape, vector< Shape > &shapes, float maximumDistance  );
    cv::Mat removeBlack( cv::Mat input, short nearLimit, short farLimit );
	void update();
	void draw();
    
    OpenNI::DeviceRef mDevice;
    OpenNI::DeviceManagerRef mDeviceManager;
    
    ci::Surface8u mSurface;
    ci::Surface8u mSurfaceDepth;
    ci::Surface8u mSurfaceBlur;
    ci::Surface8u mSurfaceSubtract;
    gl::TextureRef mTexture;
    gl::TextureRef mTextureDepth;
    
    cv::Mat mPreviousFrame;
    cv::Mat mBackground;
    
    params::InterfaceGlRef mParams;
    double mThresh;
    double mMaxVal;
    short mNearLimit;
    short mFarLimit;
  private:
    typedef vector< vector<cv::Point > > ContourVector;
    ContourVector mContours;
    ContourVector mApproxContours;
    int mStepSize;
    int mBlurAmount;
    int shapeUID;
    
    cv::Mat mInput;
    cv::vector<cv::Vec4i> mHierarchy;
    vector<Shape> mShapes;
    vector<Shape> mTrackedShapes;
};

void MotionTrackingTestApp::setup(){
    mDeviceManager = OpenNI::DeviceManager::create();
    
    shapeUID = 0;
    mTrackedShapes.clear();
    
    if( mDeviceManager->isInitialized() ){
        try{
            mDevice = mDeviceManager->createDevice( OpenNI::DeviceOptions().enableColor() );
        } catch( OpenNI::ExcDeviceNotAvailable ex) {
            console() << ex.what() << endl;
            quit();
            return;
        }
        
        if( mDevice ){
            mDevice->connectDepthEventHandler( &MotionTrackingTestApp::onDepth, this );
            mDevice->connectColorEventHandler( &MotionTrackingTestApp::onColor, this );
         //   mBackground = cv::Mat( 240,320, CV_16UC1 );
         //   mPreviousFrame = cv::Mat( 240,320, CV_16UC1 );
            mDevice->start();
        }
    }
    
    mThresh = 0.0;
    mMaxVal = 255.0;
    mNearLimit = 30;
    mFarLimit = 4000;
    
    mParams = params::InterfaceGl::create("Threshold", Vec2i( 255, 200 ) );
    mParams->addParam("Thresh", &mThresh, "min=0.0f max=255.0f step=1.0 keyIncr=a keyDecr=s");
    mParams->addParam("Maxval", &mMaxVal, "min=0.0f max=255.0f step=1.0 keyIncr=q keyDecr=w");
    //mParams->addParam( "Black near", &mNearLimit, "min=10 max=100 step=1 keyIncr=t keyDecr=y" );
//    mParams->addParam( "Black far", &mFarLimit, "min=200 max=1000 step=1 keyIncr=g keyDecr=h" );
    mStepSize = 10;
    mBlurAmount = 10;
}

void MotionTrackingTestApp::prepareSettings( Settings* settings ){
    settings->setFrameRate( 60.0f );
    settings->setWindowSize( 800, 800 );
}

//void MotionTrackingTestApp::keyDown( KeyEvent event ){
//    mPreviousFrame.copyTo( mBackground );
//    mBackground = removeBlack( mBackground, mNearLimit, mFarLimit );
//}

void MotionTrackingTestApp::onDepth( openni::VideoFrameRef frame, const OpenNI::DeviceOptions& deviceOptions){
    mInput = toOcv( OpenNI::toChannel16u( frame ) );
    
    cv::Mat withoutBlack;
    withoutBlack = removeBlack( mInput, mNearLimit, mFarLimit );
    
    cv::Mat eightBit;
    cv::Mat thresh;
    
    // convert to RGB color space, with some compensation
    withoutBlack.convertTo( eightBit, CV_8UC3, 0.1/1.0  );
    cv::bitwise_not(eightBit, eightBit);
    
    mContours.clear();
    mApproxContours.clear();
    cv::threshold( eightBit, thresh, mThresh, mMaxVal, CV_8U );
    cv::findContours( thresh, mContours, mHierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE );
    
    vector<cv::Point> approx;
    // approx number of points per contour
    for( int i=0; i<mContours.size(); i++ ) {
        cv::approxPolyDP(mContours[i], approx, 3, true );
        mApproxContours.push_back( approx );
    }
    
    // get data that we can later compare
    mShapes.clear();
    mShapes = getEvaluationSet( mApproxContours, 75, 100000 );
    
    // find the nearest match for each shape
    for( int i = 0; i<mTrackedShapes.size(); i++ ){
        Shape* nearestShape = findNearestMatch( mTrackedShapes[i], mShapes, 5000 );
        
        if( nearestShape != NULL){
            // update our tracked contour
            // last frame seen
            nearestShape->matchFound = true;
            mTrackedShapes[i].centroid = nearestShape->centroid;
            mTrackedShapes[i].lastFrameSeen = ci::app::getElapsedFrames();
            mTrackedShapes[i].hull.clear();
            mTrackedShapes[i].hull = nearestShape->hull;
        }
    }
    
    // if shape->matchFound is false, add it as a new shape
    for( int i = 0; i<mShapes.size(); i++ ){
        if( mShapes[i].matchFound == false ){
            mShapes[i].ID = shapeUID;
            mShapes[i].lastFrameSeen = ci::app::getElapsedFrames();
            mTrackedShapes.push_back( mShapes[i]);
            shapeUID++;
//            std::cout << "adding a new tracked shape with ID: " << mShapes[i].ID << std::endl;
        }
    }
    
    // if we didnt find a match for x frames, delete the tracked shape
    for( vector<Shape>::iterator it=mTrackedShapes.begin(); it!=mTrackedShapes.end(); ){
//        std::cout << "tracked shapes size: " << mTrackedShapes.size() << std::endl;
        if( ci::app::getElapsedFrames() - it->lastFrameSeen > 20 ){
//            std::cout << "deleting shape with ID: " << it->ID << std::endl;
            it = mTrackedShapes.erase(it);
        } else {
            ++it;
        }
    }
    
    cv::Mat gray8Bit;
    withoutBlack.convertTo( gray8Bit, CV_8UC3, 0.1/1.0  );
    
    mSurfaceDepth = Surface8u( fromOcv( mInput  ) );
    mSurfaceBlur = Surface8u( fromOcv( withoutBlack ) );
    mSurfaceSubtract = Surface8u( fromOcv( eightBit ) );
}

void MotionTrackingTestApp::onColor(openni::VideoFrameRef frame, const OpenNI::DeviceOptions& deviceOptions){
    mSurface = OpenNI::toSurface8u( frame );
    cv::Mat mInput( toOcv( OpenNI::toSurface8u( frame ), 0 ) );
}

vector< Shape > MotionTrackingTestApp::getEvaluationSet( vector< vector<cv::Point> > rawContours, int minimalArea, int maxArea ){
    vector< Shape > vec;
    for ( vector< cv::Point > &c : rawContours )
    {
        // create a matrix from the contour
        cv::Mat matrix = cv::Mat( c );
        
        // extract data from contour
        cv::Scalar center = mean( matrix );
        double area = cv::contourArea( matrix );
        
        // reject it if too small
        if ( area < minimalArea )
            continue;
        
        // reject it if too big
        if ( area > maxArea )
            continue;
        
        // store data
        Shape shape;
        shape.area = area;
        shape.centroid = cv::Point(center.val[0], center.val[1]);
        
        // convex hull is the polygon enclosing the contour
        shape.hull = c;
        console() << "shape points: " << shape.hull.size() << endl;
        shape.matchFound = false;
        vec.push_back( shape );
    }
    return vec;
}

Shape* MotionTrackingTestApp::findNearestMatch( Shape trackedShape, vector< Shape > &shapes, float maximumDistance  )
{
    Shape* closestShape = NULL;
    float nearestDist = 1e5;
    if ( shapes.empty() ){
        return NULL;
    }
    
    for ( Shape &candidate : shapes )
    {
        // find dist between the center of the contour and the shape
        cv::Point distPoint = trackedShape.centroid - candidate.centroid;
        float dist = cv::sqrt( distPoint.x*distPoint.x + distPoint.y*distPoint.y );
        if ( dist > maximumDistance )
            continue;
        
        if ( candidate.matchFound )
            continue;
        
        if ( dist < nearestDist )
        {
            nearestDist = dist;
            closestShape = &candidate;
        }
    }
    return closestShape;
}

cv::Mat MotionTrackingTestApp::removeBlack( cv::Mat input, short nearLimit, short farLimit )
{
    int counter = 0;
    
    for( int y=0; y<input.rows; y++ ){
        for( int x=0; x<input.cols; x++ ){
//            short color = input.at<int>(y,x);
  //          std::cout << "color " << color << std::endl;
            if( input.at<short>(y,x) < nearLimit || input.at<short>(y,x) > farLimit){
                input.at<short>(y,x) = 4000;
//                std::cout << "color after " << input.at<short>(y,x) << std::endl;
            }
        }
    }
//    console() << "counter " << counter << endl;
    return input;
}

void MotionTrackingTestApp::update()
{
}

void MotionTrackingTestApp::draw()
{
   // gl::setViewport( getWindowBounds() );
    // clear out the window with black
	gl::clear( Color( 1, 1, 1 ) );
    
//    if( mSurface ){
//        if( mTexture ){
//            mTexture->update( mSurface );
//        } else {
//            mTexture = gl::Texture::create( mSurface );
//        }
//        gl::draw( mTexture, mTexture->getBounds(), getWindowBounds() );
//    }
    
    if( mSurfaceDepth ){
        if( mTextureDepth ){
            mTextureDepth->update( Channel32f( mSurfaceDepth ) );
        } else {
            mTextureDepth = gl::Texture::create( Channel32f( mSurfaceDepth ) );
        }
        gl::color( Color::white() );
        gl::draw( mTextureDepth, mTextureDepth->getBounds() );
    }
    gl::pushMatrices();
    gl::translate( Vec2f( 320, 0 ) );
    if( mSurfaceBlur ){
        if( mTextureDepth ){
            mTextureDepth->update( Channel32f( mSurfaceBlur ) );
        } else {
            mTextureDepth = gl::Texture::create( Channel32f( mSurfaceBlur ) );
        }
        gl::draw( mTextureDepth, mTextureDepth->getBounds() );
    }
    gl::translate( Vec2f( 0, 240 ) );
    if( mSurfaceSubtract ){
        if( mTextureDepth ){
            mTextureDepth->update( Channel32f( mSurfaceSubtract ) );
        } else {
            mTextureDepth = gl::Texture::create( Channel32f( mSurfaceSubtract ) );
        }
        gl::draw( mTextureDepth, mTextureDepth->getBounds() );
    }
    gl::translate( Vec2f( -320, 0 ) );
    for( ContourVector::iterator iter = mContours.begin(); iter != mContours.end(); ++iter ){
        glBegin( GL_LINE_LOOP );
            for( vector< cv::Point >::iterator pt = iter->begin(); pt != iter->end(); ++pt ){
                gl::color( Color( 1.0f, 0.0f, 0.0f ) );
                gl::vertex( fromOcv( *pt ) );
            }
            glEnd();
    }
    gl::translate( Vec2f( 0, 240 ) );
    for( int i=0; i<mTrackedShapes.size(); i++){
        glBegin( GL_POINTS );
        for( int j=0; j<mTrackedShapes[i].hull.size(); j++ ){
           gl::color( Color( 1.0f, 0.0f, 0.0f ) );
           gl::vertex( fromOcv( mTrackedShapes[i].hull[j] ) );
        }
        glEnd();
    }
    gl::popMatrices();
    mParams->draw();
}

CINDER_APP_NATIVE( MotionTrackingTestApp, RendererGl )
