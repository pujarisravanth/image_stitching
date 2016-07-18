#include <iostream>
#include <fstream>
#include <string>
#include "opencv2/opencv_modules.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/stitching/detail/autocalib.hpp"
#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/timelapsers.hpp"
#include "opencv2/stitching/detail/camera.hpp"
#include "opencv2/stitching/detail/exposure_compensate.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"
#include "opencv2/stitching/detail/util.hpp"
#include "opencv2/stitching/detail/warpers.hpp"
#include "opencv2/stitching/warpers.hpp"

using namespace std;
using namespace cv;
using namespace cv::detail;

bool try_cuda = false;
float match_conf = 0.4f;

int main(int argc, char* argv[]) {

	Mat img1 = imread(argv[1]);
	Mat img2 = imread(argv[2]);

	Ptr<FeaturesFinder> finder;
	finder = makePtr<SurfFeaturesFinder>();

	ImageFeatures features1, features2;

	(*finder)(img1, features1);
	(*finder)(img2, features2);

	features1.img_idx = 1;
	features2.img_idx = 2;
	
	finder->collectGarbage();

	cout << "No. of keypoints1 : " << features1.keypoints.size() << endl ;
	cout << "No. of keypoints2 : " << features2.keypoints.size() << endl ;

	Mat img_kp_1, img_kp_2;

	drawKeypoints(img1, features1.keypoints, img_kp_1, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
	drawKeypoints(img2, features2.keypoints, img_kp_2, Scalar::all(-1), DrawMatchesFlags::DEFAULT );

	imwrite("img_kp1.jpg", img_kp_1);
	imwrite("img_kp2.jpg", img_kp_2);

	MatchesInfo matches;
	
	BestOf2NearestMatcher matcher(try_cuda, match_conf);
	matcher(features1, features2, matches);
	matcher.collectGarbage();

	cout << matches.confidence << endl;
	cout << matches.H << endl;
	cout << matches.num_inliers << endl;
	cout << matches.matches.size() << endl;
	cout << matches.src_img_idx << endl;
	cout << matches.dst_img_idx << endl;

	float confidence = matches.num_inliers / (8 + 0.3 * matches.matches.size());
	cout << confidence << endl;

	Mat img_matches;
    drawMatches( img1, features1.keypoints, img2, features2.keypoints,
    		matches.matches, img_matches, Scalar::all(-1), Scalar::all(-1),
                 vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

    imwrite( "Good_Matches.jpg", img_matches );

	return 0; }