#include "Application.h"

// contains all the method definitions for the Application class

// Member functions
// Non-static
void Application::readImages(vector<Mat>& inputImages) {
	// check her to see if all fields have been initialized
	if(this->N != 0 && this->fileType.length() != 0 && this->fileName.length() != 0 && this->resourcePath.length() != 0) {

		// first check if resource path has last character as \
		// else append one
		if(resourcePath[resourcePath.length()-1] != '\\') {
			resourcePath = resourcePath + string("\\");
		}
		
		// now logic for reading off images
		for(int i = this->startIndex; i <= this->endIndex; i++) {
			Mat image;
			stringstream fileno(stringstream::in|stringstream::out);
			//create the image path string and read it into file
			fileno<<i;
			string filepath = resourcePath + fileName + fileno.str() + fileType;
			//cout<<filepath<<endl;
			image = imread(filepath);
			inputImages.push_back(image);
		}
	} else {
		cerr<<"\n APPLICATION NOT INITIALIZED PROPERLY";
		cerr<<"\n Please use setter methods or construct application properly before proceeding.";
	}
}

// given a set of inputImages, this handles all the logic required to display and get matching points and stores them to a file in the format
// FILENAME : srcImage+DestImage.mpt
// file is stored in outputDir
// NOTE : Its upto user to send in resized images or whatever. --> to do so we can use application provided static function.. see app headerile
void Application::matchPoints(vector<Mat>& inputImages, string outputDir, bool computeHomographies, vector<Mat>* homographyMatrices) {


	namedWindow("source");
	namedWindow("destination");
	Message* srcMessage;
	Message* destMessage;
	vector<Point2f> srcPoints;
	vector<Point2f> destPoints;
	Mat src;
	Mat dest;
	stringstream srcIndex;
	stringstream destIndex;
	this->matchPointDir = outputDir;
	
	// for 1 till middle image calculate H matrices
	for(int i = startIndex; i < centerIndex; i++) {
		
		inputImages.at(i - startIndex).copyTo(src);
		inputImages.at(i+1 - startIndex).copyTo(dest);
		srcPoints = vector<Point2f>(0);
		destPoints = vector<Point2f>(0);
		
		srcMessage = new Message(src, "source", &srcPoints);
		destMessage = new Message(dest, "destination", &destPoints);

		// set mouse call back methods
		setMouseCallback("source", onMouseMessage, (void*)srcMessage);
		setMouseCallback("destination", onMouseMessage, (void*)destMessage);

		// show windows and waitkey
		imshow("source", src);
		imshow("destination", dest);
		waitKey(0);

		// create object of matchingpoints and write to file
		srcIndex<<i;
		destIndex<<i+1;
		
		if(computeHomographies) {
			Mat H = Application::computeHomography(srcPoints, destPoints);
			homographyMatrices->push_back(H);
			HomographyFile hf(outputDir, fileName + srcIndex.str(), fileName + destIndex.str(), H);
			hf.writeFile();
		}
		
		MatchingPoints mp(outputDir, fileName + srcIndex.str(), fileName + destIndex.str(), srcPoints, destPoints);
		mp.writeToFile();
		srcIndex.str("");
		destIndex.str("");
		// free up memory
		delete srcMessage;
		delete destMessage;
	}
	// for middle till end calculate H matrices
	for(int i = centerIndex; i < endIndex; i++) {

		// this side the 
		// i = dest image
		// i+1 = src image
		inputImages.at(i+1 - startIndex).copyTo(src);
		inputImages.at(i - startIndex).copyTo(dest);
		srcPoints = vector<Point2f>(0);
		destPoints = vector<Point2f>(0);
		
		srcMessage = new Message(src, "source", &srcPoints);
		destMessage = new Message(dest, "destination", &destPoints);

		// set mouse call back methods
		setMouseCallback("source", onMouseMessage, (void*)srcMessage);
		setMouseCallback("destination", onMouseMessage, (void*)destMessage);

		// show windows and waitkey
		imshow("source", src);
		imshow("destination", dest);
		waitKey(0);

		// create object of matchingpoints and write to file
		srcIndex<<i+1;
		destIndex<<i;
		
		if(computeHomographies) {
			Mat H = computeHomography(srcPoints, destPoints);
			homographyMatrices->push_back(H);
			HomographyFile hf(outputDir, fileName + srcIndex.str(), fileName + destIndex.str(), H);
			hf.writeFile();
		}

		MatchingPoints mp(outputDir, fileName + srcIndex.str(), fileName + destIndex.str(), srcPoints, destPoints);
		mp.writeToFile();

		srcIndex.str("");
		destIndex.str("");
		// free up memory
		delete srcMessage;
		delete destMessage;
	}
}

// if an application has been initialized, then if matching points have already been found,
// we can read them using the following method
void Application::readMatchingPoints(vector<vector<Point2f>>& outputSrcPoints, vector<vector<Point2f>>& outputDestPoints, string matchPointDir) {

	if(matchPointDir[matchPointDir.length() - 1] != '\\') {
		matchPointDir = matchPointDir + string("\\");
	}
	stringstream srcIndex, destIndex;
	for(int i = startIndex; i < centerIndex; i++) {
		MatchingPoints p;
		//form the filename first and then we can read off the points
		srcIndex<<i;
		destIndex<<i+1;
		string file = fileName + srcIndex.str() + fileName + destIndex.str() + string(".mpt");
		p.readFile(file, matchPointDir);

		// put contents into the src and destination points
		outputSrcPoints.push_back(p.getSrcPoints());
		outputDestPoints.push_back(p.getDestPoints());

		srcIndex.str("");
		destIndex.str("");
	}

	// for the other images
	for(int i = centerIndex; i < endIndex; i++) {
		MatchingPoints p;
		srcIndex<<i+1;
		destIndex<<i;
		string file = fileName + srcIndex.str() + fileName + destIndex.str() + string(".mpt");
		p.readFile(file, matchPointDir);

		// put contents into the src and destination points
		outputSrcPoints.push_back(p.getSrcPoints());
		outputDestPoints.push_back(p.getDestPoints());

		srcIndex.str("");
		destIndex.str("");
	}

}

// initialized an application, we can read off the homography matrices if the already exist
void Application::readHomographyMatrices(vector<Mat>& outputHomographyMatrices, string matchPointDir) {

	if(matchPointDir[matchPointDir.length() - 1] != '\\') {
		matchPointDir = matchPointDir + string("\\");
	}
	stringstream srcIndex, destIndex;
	for(int i = startIndex; i < centerIndex; i++) {
		HomographyFile f;
		int src = i;
		int dest = i+1;
		srcIndex<<src;
		destIndex<<dest;
		string file = fileName + srcIndex.str() + fileName + destIndex.str() + string(".hmg");
		f.readFile(file, matchPointDir);
		outputHomographyMatrices.push_back(f.getHomographyMatrix());
		srcIndex.str("");
		destIndex.str("");
	}
	for(int i = centerIndex; i < endIndex; i++) {
		HomographyFile f;
		int src = i+1;
		int dest = i;
		srcIndex<<src;
		destIndex<<dest;
		string file = fileName + srcIndex.str() + fileName + destIndex.str() + string(".hmg");
		f.readFile(file, matchPointDir);
		outputHomographyMatrices.push_back(f.getHomographyMatrix());
		srcIndex.str("");
		destIndex.str("");
	}
}

// static methods

// Bilinear interpolation
// Returns a scalar of the color of the interpolated point from the input image
Scalar Application::Bilerp(Mat& src, Point2d point) {

	double x = point.x;
	double y = point.y;
	int i = static_cast<int>(floor(x));
	int j = static_cast<int>(floor(y));
	
	double a = x - i;
	double b = y - j;

	Mat weights = (Mat_<double>(4, 1) << (1-a)*(1-b), a*(1-b), a*b, (1-a)*b );

	Mat_<Vec3b> I = src;
	Mat colors(3, 4, CV_64F, Scalar(0));
	
	// create a matrix of red, green and blue channels 
	// 3 * 4 matrix
	if(x >= 0 && x <= static_cast<double>(src.cols - 2) && y >= 0 && y <= static_cast<double>(src.rows - 2)) {
		colors = (Mat_<double>(3, 4) << static_cast<double>(I(j, i)[0]), static_cast<double>(I(j, i+1)[0]), static_cast<double>(I(j+1, i+1)[0]), static_cast<double>(I(j+1, i)[0]),
											static_cast<double>(I(j, i)[1]), static_cast<double>(I(j, i+1)[1]), static_cast<double>(I(j+1, i+1)[1]), static_cast<double>(I(j+1, i)[1]),
											static_cast<double>(I(j, i)[2]), static_cast<double>(I(j, i+1)[2]), static_cast<double>(I(j+1, i+1)[2]), static_cast<double>(I(j+1, i)[2]));
	} else if(x == static_cast<double>(src.cols - 1) && static_cast<double>(y < src.rows - 2)) {
		colors = (Mat_<double>(3, 4) << static_cast<double>(I(j, i)[0]), static_cast<double>(I(j, i-1)[0]), static_cast<double>(I(j+1, i-1)[0]), static_cast<double>(I(j+1, i)[0]),
											static_cast<double>(I(j, i)[1]), static_cast<double>(I(j, i-1)[1]), static_cast<double>(I(j+1, i-1)[1]), static_cast<double>(I(j+1, i)[1]),
											static_cast<double>(I(j, i)[2]), static_cast<double>(I(j, i-1)[2]), static_cast<double>(I(j+1, i-1)[2]), static_cast<double>(I(j+1, i)[2]));
	} else if(static_cast<double>(x < src.cols - 2) && static_cast<double>(y == src.rows - 1)) {
		colors = (Mat_<double>(3, 4) << static_cast<double>(I(j, i)[0]), static_cast<double>(I(j, i+1)[0]), static_cast<double>(I(j-1, i+1)[0]), static_cast<double>(I(j-1, i)[0]),
											static_cast<double>(I(j, i)[1]), static_cast<double>(I(j, i+1)[1]), static_cast<double>(I(j-1, i+1)[1]), static_cast<double>(I(j-1, i)[1]),
											static_cast<double>(I(j, i)[2]), static_cast<double>(I(j, i+1)[2]), static_cast<double>(I(j-1, i+1)[2]), static_cast<double>(I(j-1, i)[2]));
	} else if(static_cast<double>(x == src.cols - 1) && static_cast<double>(y == src.rows - 1)) {
		colors = (Mat_<double>(3, 4) << static_cast<double>(I(j, i)[0]), static_cast<double>(I(j, i-1)[0]), static_cast<double>(I(j-1, i-1)[0]), static_cast<double>(I(j-1, i)[0]),
											static_cast<double>(I(j, i)[1]), static_cast<double>(I(j, i-1)[1]), static_cast<double>(I(j-1, i-1)[1]), static_cast<double>(I(j-1, i)[1]),
											static_cast<double>(I(j, i)[2]), static_cast<double>(I(j, i-1)[2]), static_cast<double>(I(j-1, i-1)[2]), static_cast<double>(I(j-1, i)[2]));
	}


	Mat resColor = colors * weights;
	
	uchar ur = static_cast<uchar>(floor(resColor.at<double>(2, 0)));
	uchar ug = static_cast<uchar>(floor(resColor.at<double>(1, 0)));
	uchar ub = static_cast<uchar>(floor(resColor.at<double>(0, 0)));
	return Scalar(ur, ug, ub);
}

// This method computes the bounding box for the image using the given homographyMatrix
Rect Application::calculateBoundingBox(Mat& image, Mat& H) {

	int width = image.cols;
	int height = image.rows;
	
	// form a matrix of points or corners
	// (0,0) --> (width-1, 0) ---> (width-1, height-1) ---> (0, height-1)
	// because zero indexing
	Mat corners = (Mat_<double>(3, 4) << 0, width-1, width-1,  0,
										 0, 0,       height-1, height-1,
										 1, 1,       1,        1);

	Mat xfmCorners = H * corners;
	// store as they come
	double xs[4];
	double ys[4];
	// bring w component to 1 of all columns
	xs[0] = xfmCorners.at<double>(0, 0) / xfmCorners.at<double>(2, 0);
	ys[0] = xfmCorners.at<double>(1, 0) / xfmCorners.at<double>(2, 0);
	xs[1] = xfmCorners.at<double>(0, 1) / xfmCorners.at<double>(2, 1);
	ys[1] = xfmCorners.at<double>(1, 1) / xfmCorners.at<double>(2, 1);
	xs[2] = xfmCorners.at<double>(0, 2) / xfmCorners.at<double>(2, 2);
	ys[2] = xfmCorners.at<double>(1, 2) / xfmCorners.at<double>(2, 2);
	xs[3] = xfmCorners.at<double>(0, 3) / xfmCorners.at<double>(2, 3);
	ys[3] = xfmCorners.at<double>(1, 3) / xfmCorners.at<double>(2, 3);

	// now find the min-x, min-y and max-x and max-y
	double min_x = std::numeric_limits<double>::max();
	double min_y = std::numeric_limits<double>::min();
	double max_x = -1, max_y = -1;
	for(int i = 0; i < 4; i++) {
		// min code
		if(xs[i] < min_x) min_x = xs[i];
		if(ys[i] < min_y) min_y = ys[i];

		// max code
		if(xs[i] > max_x) max_x = xs[i];
		if(ys[i] > max_y) max_y = ys[i];
	}
	
	// return bounding box
	return Rect(static_cast<int>(floor(min_x)), static_cast<int>(floor(min_y)), static_cast<int>(floor(max_x - min_x)), static_cast<int>(floor(max_y - min_y)));
}

// Get the transformed Corners to get a sense of how much area the image would occupy
Mat Application::calculateXfmImageCorners(Mat& image, Mat& H) {

	int width = image.cols;
	int height = image.rows;
	
	// form a matrix of points or corners
	// (0,0) --> (width-1, 0) ---> (width-1, height-1) ---> (0, height-1)
	// because zero indexing
	Mat corners = (Mat_<double>(3, 4) << 0, width-1, width-1,  0,
										 0, 0,       height-1, height-1,
										 1, 1,       1,        1);

	Mat xfmCorners = H * corners;
	// store as they come
	double xs[4];
	double ys[4];
	// bring w component to 1 of all columns
	xs[0] = xfmCorners.at<double>(0, 0) / xfmCorners.at<double>(2, 0);
	ys[0] = xfmCorners.at<double>(1, 0) / xfmCorners.at<double>(2, 0);
	xs[1] = xfmCorners.at<double>(0, 1) / xfmCorners.at<double>(2, 1);
	ys[1] = xfmCorners.at<double>(1, 1) / xfmCorners.at<double>(2, 1);
	xs[2] = xfmCorners.at<double>(0, 2) / xfmCorners.at<double>(2, 2);
	ys[2] = xfmCorners.at<double>(1, 2) / xfmCorners.at<double>(2, 2);
	xs[3] = xfmCorners.at<double>(0, 3) / xfmCorners.at<double>(2, 3);
	ys[3] = xfmCorners.at<double>(1, 3) / xfmCorners.at<double>(2, 3);

	// form a new matrix with these values
	Mat rXfmCorners = (Mat_<double>(2, 4) << xs[0], xs[1], xs[2], xs[3],
											ys[0], ys[1], ys[2], ys[3]);

	return rXfmCorners;
}

// This method computes the sequence of homography matrices required to move it to the target
// target is essentially the central image in the composite image
Mat Application::getFinalHomographyMatrix(vector<Mat>& H, int center_index, int src_index) {

	Mat finalHomography;

	/*
	Logic:
	Assume images are as in sequence
		Image				H(srcdest)
		 0					01
		 1					12
		 2					23
		 3					34
middle-> 4					54						
		 5					65
		 6					76	
		 7

		 so if src is above middle, then we multiply from mat(loc) till (loc < middle)
		 else the other way around..
	*/
	if(src_index < center_index) {
		// move from left to center
		// starting matrix.. multiply till central and previous matrix
		finalHomography = H.at(src_index);			
		for(int i = src_index+1; i < center_index; i++) {
			finalHomography = finalHomography * H.at(i);
		}
	} else if(src_index > center_index) {
		// move from right to center
		finalHomography = H.at(src_index - 1);
		for(int i = src_index - 2; i >= center_index; i--) {
			finalHomography = finalHomography * H.at(i);
		}
	}
	return finalHomography;
}

// One time calculation of homography matrices for all images to the central image
vector<Mat> Application::calculateAllFinalHomographyMatrices(vector<Mat>& H, int center_index) {
	vector<Mat> finalHomographyMatrices;
	for(int i = 0; i < center_index && i < H.size(); i++) {
		//int src_index = i;
		//Mat temp;
		//temp = Application::getFinalHomographyMatrix(H, center_index, i);
		//
		finalHomographyMatrices.push_back(getFinalHomographyMatrix(H, center_index, i));
	}
	for(int i = H.size() - 1; i >= center_index; i--) {
		finalHomographyMatrices.push_back(getFinalHomographyMatrix(H, center_index, i));
	}
	return finalHomographyMatrices;
}

// This method is used to compute the cylindrical transform of the input image so that it can be stitched in a good manner
Mat Application::getCylindricalProjection(Mat& inputImage, double focal) {

	Mat output(inputImage.size(), inputImage.type());
	int xc = inputImage.cols/2;
	int yc = inputImage.rows/2;
	double f = focal;

	for (int y=0;y<inputImage.rows;y++){
		for(int x=0;x<inputImage.cols;x++){
			double theta = (x-xc)/f;
			double h = (y-yc)/f;
			double xcap = sin(theta);
			double ycap = h;
			double zcap = cos(theta);
			double xn = xcap/zcap;
			double yn = ycap/zcap;
			double r = pow(xn,2)+pow(yn,2);
			double xd = xn;
			double yd = yn;
			int ximg = static_cast<int>(floor(f*xd + xc));
			int yimg = static_cast<int>(floor(f*yd + yc));
			if(ximg > 0 && ximg <= inputImage.cols && yimg > 0 && yimg <= inputImage.rows) {
				output.at<Vec3b>(y,x)[0] = inputImage.at<Vec3b>(yimg,ximg)[0];
				output.at<Vec3b>(y,x)[1] = inputImage.at<Vec3b>(yimg,ximg)[1];
				output.at<Vec3b>(y,x)[2] = inputImage.at<Vec3b>(yimg,ximg)[2];
			}
		}
	}

	return output;
}

// Mouse message functions
// This function can be used to click on images and collecting information about points clicked
// and also displays dots over points clicked with index values
void Application::onMouseMessage(int event, int x, int y, int flags, void* msg) {
	Message* tMsg = (Message*)msg;
	stringstream point(stringstream::in|stringstream::out);
	int pointCnt;			// indicates which point is being drawn
	switch(event) {
	case CV_EVENT_LBUTTONDOWN:
		// do the destination window please
		tMsg->matchingPoints->push_back(Point2f(static_cast<float>(x), static_cast<float>(y)));		// add point
		pointCnt = tMsg->matchingPoints->size();
		point<<pointCnt;
		if(strcmp(tMsg->window_name, "destination") == 0) {
			// draw a circle on point and mark the index of point
			circle(tMsg->image, Point(x, y),  5, Scalar(0, 0, 255), -1);
			putText(tMsg->image, point.str(), Point(x+5, y), 1, 1.2, Scalar(255, 255, 0));
			imshow(tMsg->window_name, tMsg->image);
		} else if(strcmp(tMsg->window_name, "source") == 0) {
			// on the source please
			circle(tMsg->image, Point(x, y),  5, Scalar(0, 255, 0), -1);
			putText(tMsg->image, point.str(), Point(x+5, y), 1, 1.2, Scalar(255, 255, 0));
			imshow(tMsg->window_name, tMsg->image);
		}
	}
}

Mat Application::computeHomography(vector<Point2f>& src, vector<Point2f>& dest, Method m) {
	
	if(m == MY_METHOD) {
		// first my method --> DLT implementation	
		// given two images, we calculate the homography matrix.
		Mat homographyMatrix(3, 3, CV_64F);
		// Ah = 0;
		// first we create A matrix
		int N1 = dest.size();
		int N2 = src.size();
		if(N1 != N2) {
			cerr<<"Error. Unequal number of matching points";
			exit(-1);
		}

		Mat A(2*N1, 9, CV_64F);		// 2N x 9 is the dimension of the matrix
		int w = 1, wd = 1;			// w and w-dash
		vector<Point2f>::iterator iter1 = dest.begin();
		vector<Point2f>::iterator iter2 = src.begin();
		int row = 0;
	
		while(iter1 != dest.end() && iter2 != src.end()) {
			int col = 0;
			// get individual x and y per col
			double x = (*iter2).x; double y = (*iter2).y;
			double xd = (*iter1).x; double yd = (*iter1).y;
			{
				// even rows
				A.at<double>(row, 0) = 0.0; A.at<double>(row, 1) = 0.0; A.at<double>(row, 2) = 0.0;
				A.at<double>(row, 3) = -x; A.at<double>(row, 4) = -y; A.at<double>(row, 5) = -1.0f;
				A.at<double>(row, 6) = x * yd; A.at<double>(row, 7) = y * yd; A.at<double>(row, 8) = yd;
			}  {
				// odd rows
				A.at<double>(row + 1, 0) = x; A.at<double>(row + 1, 1) = y; A.at<double>(row + 1, 2) = 1.0;
				A.at<double>(row + 1, 3) = 0.0; A.at<double>(row + 1, 4) = 0.0; A.at<double>(row + 1, 5) = 0.0;
				A.at<double>(row + 1, 6) = -xd * x; A.at<double>(row + 1, 7) = -xd * y; A.at<double>(row + 1, 8) = -xd;
			}
			row += 2;		// we are creating two rows per point. hence increment by 2
			iter1++;
			iter2++;
		}
		// now that we have the matrix A, we can just perform SVD and then get the column on V corresponding to minimal D
		// A = UDV'
		Mat u, d, v;
		SVD::compute(A, d, u, v);
		// last row of v since v is actually v'
		transpose(v, v);
		int lrow = 0;			// zero indexing
		int lcols = v.cols - 1;	
		for(int i = 0; i < 3; i++) {
			for(int j = 0; j < 3; j++) {
				homographyMatrix.at<double>(i, j) = v.at<double>(lrow++, lcols);
			}
		}
		double dw = homographyMatrix.at<double>(2, 2);
		homographyMatrix = homographyMatrix / dw;
		return homographyMatrix;
	} else {
		// use opencv with ransac
		return findHomography(src, dest, CV_RANSAC);
	}
}

Mat Application::computeHomographyNormalized(vector<Point2f>& src, vector<Point2f>& dest, Point2i srcDim, Point2i destDim, Method m) {

	// Pre checks
	Mat homographyMatrix(3, 3, CV_64F);
	// Ah = 0;
	// first we create A matrix
	int N1 = dest.size();
	int N2 = src.size();
	if(N1 != N2) {
		cerr<<"Error. Unequal number of matching points";
		exit(-1);
	}

	Mat A(2*N1, 9, CV_64F);		// 2N x 9 is the dimension of the matrix
	int w = 1, wd = 1;			// w and w-dash
	
	// homography matrix by normalization of points
	Mat Tnorm = (Mat_<double>(3, 3) << static_cast<double>(srcDim.x + srcDim.y), 0, static_cast<double>(srcDim.x/2),
									  0, static_cast<double>(srcDim.x + srcDim.y), static_cast<double>(srcDim.y/2),
									  0, 0, 1);
	invert(Tnorm, Tnorm);		// actually Tnorm is the above matrices' inverse

	Mat TnormDash = (Mat_<double>(3, 3) << static_cast<double>(destDim.x + srcDim.y), 0, static_cast<double>(destDim.x/2),
									  0, static_cast<double>(destDim.x + destDim.y), static_cast<double>(destDim.y/2),
									  0, 0, 1);

	invert(TnormDash, TnormDash);

	// now multiply all points by their respective Tnorm matrices
	vector<Point2d> modifiedSrc;
	vector<Point2d> modifiedDest;

	for(int i = 0; i < src.size(); i++) {
		Mat pt = (Mat_<double>(3, 1) << static_cast<double>(src.at(i).x), static_cast<double>(src.at(i).y), 1);
		Mat temp = Tnorm * pt;
		temp.at<double>(0, 0) /= temp.at<double>(2, 0);
		temp.at<double>(1, 0) /= temp.at<double>(2, 0);
		modifiedSrc.push_back(Point2d(temp.at<double>(0, 0), temp.at<double>(1, 0)));
	}

	// now for destination points
	for(int i = 0; i < dest.size(); i++) {
		Mat pt = (Mat_<double>(3, 1) << static_cast<double>(dest.at(i).x), static_cast<double>(dest.at(i).y), 1);
		Mat temp = Tnorm * pt;
		temp.at<double>(0, 0) /= temp.at<double>(2, 0);
		temp.at<double>(1, 0) /= temp.at<double>(2, 0);
		modifiedDest.push_back(Point2d(temp.at<double>(0, 0), temp.at<double>(1, 0)));
	}

	// Proceed ahead with DLT method
	vector<Point2d>::iterator iter1 = modifiedDest.begin();
	vector<Point2d>::iterator iter2 = modifiedSrc.begin();
	int row = 0;
	
	while(iter1 != modifiedDest.end() && iter2 != modifiedSrc.end()) {
		int col = 0;
		// get individual x and y per col
		double x = (*iter2).x; double y = (*iter2).y;
		double xd = (*iter1).x; double yd = (*iter1).y;
		{
			// even rows
			A.at<double>(row, 0) = 0.0; A.at<double>(row, 1) = 0.0; A.at<double>(row, 2) = 0.0;
			A.at<double>(row, 3) = -x; A.at<double>(row, 4) = -y; A.at<double>(row, 5) = -1.0f;
			A.at<double>(row, 6) = x * yd; A.at<double>(row, 7) = y * yd; A.at<double>(row, 8) = yd;
		}  {
			// odd rows
			A.at<double>(row + 1, 0) = x; A.at<double>(row + 1, 1) = y; A.at<double>(row + 1, 2) = 1.0;
			A.at<double>(row + 1, 3) = 0.0; A.at<double>(row + 1, 4) = 0.0; A.at<double>(row + 1, 5) = 0.0;
			A.at<double>(row + 1, 6) = -xd * x; A.at<double>(row + 1, 7) = -xd * y; A.at<double>(row + 1, 8) = -xd;
		}
		row += 2;		// we are creating two rows per point. hence increment by 2
		iter1++;
		iter2++;
	}

	Mat u, d, v;
	SVD::compute(A, d, u, v);
	// last row of v since v is actually v'
	transpose(v, v);
	int lrow = 0;			// zero indexing
	int lcols = v.cols - 1;	
	for(int i = 0; i < 3; i++) {
		for(int j = 0; j < 3; j++) {
			homographyMatrix.at<double>(i, j) = v.at<double>(lrow++, lcols);
		}
	}
	// premultiply by TnormDash inverse and postmultiply Tnorm
	invert(TnormDash, TnormDash);
	homographyMatrix = TnormDash * homographyMatrix;
	homographyMatrix = homographyMatrix * Tnorm;
	// normalize it
	double dw = homographyMatrix.at<double>(2, 2);
	homographyMatrix = homographyMatrix / dw;
	return homographyMatrix;
}

// NOTE : ITs assumed that user has if user has computed homography on resized images, then the input to these images too should be resized and only then sent
void Application::stitchPlanarPanaroma(vector<Mat>& images, vector<Mat>& homographyMatrices, int center_index, Method m) {
	// Planar Panaroma Stitching algorithm
	// we have opencv and my versions also
	cout<<"Stitching Panaroma :"<<endl;
	float percent_complete = 0.0f;
	if(m == OPENCV) {
		// Here we use warpPerspective method of opencv and then we compute the planar panaroma	
		//int left = abs(center_index - startIndex);
		//int right = abs(center_index - endIndex);
		Mat left;
		Mat right;
		Mat roi;
		for(int i = 0; i < center_index; i++) {
			// left side matrices
			Mat temp;
			Mat trans = (Mat_<double>(3, 3) << 1, 0, 820, 0, 1, 0, 0, 0, 1);
			Mat h = getFinalHomographyMatrix(homographyMatrices, center_index, i);
			h = trans * h;
			warpPerspective(images.at(i), temp, h, Size(2*800, 600));
			temp.copyTo(left, temp);
		}

		if(left.data != NULL) {
			namedWindow("left");
			imshow("left", left);
			waitKey(0);

			roi = Mat(left, Rect(800, 0, 800, 600));
			images.at(center_index).copyTo(roi);
			imshow("left", left);
			waitKey(0);
		}

		// RIght part
		for(int i = images.size() - 1; i > center_index; i--) {

			Mat temp;
			Mat trans = (Mat_<double>(3, 3) << 1, 0, 820, 0, 1, 0, 0, 0, 1);
			Mat h = getFinalHomographyMatrix(homographyMatrices, center_index, i);
			//h = trans * h;
			warpPerspective(images.at(i), temp, h, Size(2*800, 600));
			temp.copyTo(right, temp);
		}

		if(right.data != NULL) {
			namedWindow("right");
			imshow("right", right);
			waitKey(0);
			roi = Mat(right, Rect(0, 0, 800, 600));
			images.at(center_index).copyTo(roi);
			imshow("right", right);
			waitKey(0);
		}

		Mat final(600, 2400, images.at(center_index).type());
		if(left.data != NULL) {
			roi = Mat(final, Rect(0, 0, left.cols, left.rows));
			left.copyTo(roi);
		}
		if(right.data != NULL) {
			roi = Mat(final, Rect(800, 0, 1600, right.rows));
			right.copyTo(roi);
		}
		
		namedWindow("final");
		imshow("final", final);
		waitKey(0);
		imwrite(this->resourcePath + this->file.outputFileName, final);

		
	} else if(m == MY_METHOD) {
		// My method and planar panaroma
		// very slow but the code is all hand written
		// can be heavily optimized.
		
		// BLENDING :
		// GOTTA DO BLENDING!!!


		/*
		Method Desc : First we compose one full sized panaroma and then select an arbitrary origin and then for each pixel
					  we check if a pixels contribution can come from any of the N images. if so we get the color and then blend them
		*/


		// center_index will give the index of the central image
		vector<Mat> finalHMatrices;				// --------------> the final transform matrices required to bring all into one field
		vector<Mat> xfmCorners;

		// first find the total size of the final image
		int total_width = 0, total_height = 0;
		Point2d finalBoundingBoxLeft(FLT_MAX_10_EXP, FLT_MAX_10_EXP);
		Point2d finalBoundingBoxRight(-FLT_MAX_10_EXP, -FLT_MAX_10_EXP);
		//int mid_width;
		for(int i = 0; i < center_index; i++) {
		
			int src = i;
			finalHMatrices.push_back(Application::getFinalHomographyMatrix(homographyMatrices, center_index, src));
			Rect rect = Application::calculateBoundingBox(images.at(src), Application::getFinalHomographyMatrix(homographyMatrices, center_index, src));
			
			// create and grow the finalbounding box as required
			// first the left corner
			if(rect.x < finalBoundingBoxLeft.x) 
				finalBoundingBoxLeft.x = rect.x;
			if(rect.y < finalBoundingBoxLeft.y) 
				finalBoundingBoxLeft.y = rect.y;
			if(rect.x + rect.width > finalBoundingBoxRight.x)
				finalBoundingBoxRight.x = rect.x + rect.width;
			if(rect.y + rect.height > finalBoundingBoxRight.y)
				finalBoundingBoxRight.y = rect.y + rect.height;

			//if(total_height < rect.height) total_height = rect.height;
			//	total_width += rect.width;
		}
		
		// this is to take into account the central image
		Mat inv = Mat::eye(Size(3, 3), CV_64F);
		Rect rect = Application::calculateBoundingBox(images.at(center_index), inv);
		if(rect.x < finalBoundingBoxLeft.x) 
			finalBoundingBoxLeft.x = rect.x;
		if(rect.y < finalBoundingBoxLeft.y) 
			finalBoundingBoxLeft.y = rect.y;
		if(rect.x + rect.width > finalBoundingBoxRight.x)
			finalBoundingBoxRight.x = rect.x + rect.width;
		if(rect.y + rect.height > finalBoundingBoxRight.y)
			finalBoundingBoxRight.y = rect.y + rect.height;


		for(int i = images.size() - 1; i > center_index; i--) {
			int src = i;
			finalHMatrices.push_back(Application::getFinalHomographyMatrix(homographyMatrices, center_index, src));
			Rect rect = Application::calculateBoundingBox(images.at(src), Application::getFinalHomographyMatrix(homographyMatrices, center_index, src));

			if(rect.x < finalBoundingBoxLeft.x) 
				finalBoundingBoxLeft.x = rect.x;
			if(rect.y < finalBoundingBoxLeft.y) 
				finalBoundingBoxLeft.y = rect.y;
			if(rect.x + rect.width > finalBoundingBoxRight.x)
				finalBoundingBoxRight.x = rect.x + rect.width;
			if(rect.y + rect.height > finalBoundingBoxRight.y)
				finalBoundingBoxRight.y = rect.y + rect.height;

			//if(total_height < rect.height) total_height = rect.height;
			//total_width += rect.width;
		}

		// progress update
		cout<<"Calculated Final Bounding Box"<<endl;

		// add some buffer value + 1 width of center image which we have neglected so far
		total_width = static_cast<int>(floor(finalBoundingBoxRight.x - finalBoundingBoxLeft.x));
		total_height = static_cast<int>(floor(finalBoundingBoxRight.y - finalBoundingBoxLeft.y));


		// we setup an origin for our reference frame
		Mat finalImage(total_height, total_width, CV_8UC3);
		//int origin = (total_width - images.at(center_index).cols) / 2;
		int origin_x = static_cast<int>(floor(finalBoundingBoxLeft.x));
		int origin_y = static_cast<int>(floor(finalBoundingBoxLeft.y));
	
		Mat_<Vec3b> result = finalImage;
		for(int y = 0; y < total_height; y++) {
			for(int x = 0; x < total_width; x++) {
				// for each point check if that coord occurs in any of the images.
				// if so return the color of the interpolated point
				// then we can take an average of the colors returned and assign final pixel value
				
				// DEBUG TO SEE WHERE BREAKS
				//cout<<y<<x<<endl;

				int total_exists = 0;
				Scalar finalColor(0, 0, 0);
				vector<Scalar> colors(0);
				vector<double> minDistances(0);
				Mat Hinv;
				for(int i = 0; i < images.size(); i++) {
					if(i < center_index) {
						invert(finalHMatrices.at(i), Hinv);
					} else if(i == center_index) {
						Hinv = cv::Mat::eye(Size(3, 3), CV_64F);		// identity since center will always map to itself.
					} else if(i > center_index) {
						invert(finalHMatrices.at(i-1), Hinv);
					}
					// check if it exists in
					int loc_x = x + origin_x;
					int loc_y = y + origin_y;
					Mat p = (Mat_<double>(3, 1) << loc_x, loc_y, 1);
				
					Mat res = Hinv * p;
					Point2d backPoint(res.at<double>(0, 0)/res.at<double>(2, 0),
									  res.at<double>(1, 0)/res.at<double>(2, 0)); 
					
					
					
					if(Application::isWithinBounds(backPoint, images.at(i))) {
						
						if(file.blend == DISTANCE) {
							Mat xfmCorners;
							if(i < center_index) 
								xfmCorners = Application::calculateXfmImageCorners(images.at(i), homographyMatrices.at(i));
							if(i == center_index)
								xfmCorners = Application::calculateXfmImageCorners(images.at(i), inv);			// inv is identity matrix
							if(i > center_index)
								xfmCorners = Application::calculateXfmImageCorners(images.at(i), homographyMatrices.at(i-1));

							minDistances.push_back(Application::pointToRectEdges(xfmCorners, backPoint));
							colors.push_back(Application::Bilerp(images.at(i), backPoint));
						}
						if(file.blend == AVERAGING) {
							finalColor = finalColor + Application::Bilerp(images.at(i), backPoint);
							total_exists++;			// --------------> POSSIBLE BUG ?? ---> What if we dont map a point to source image at al
						}
					}
				}
			
				// find min distance in image and then assign that color
				if(file.blend == DISTANCE) {
					double maxD = FLT_MAX;
					for(int minD = 0; minD < minDistances.size(); minD++) {
						if(minDistances.at(minD) < maxD) 
						finalColor = colors.at(minD);
					}
				}

				if(file.blend == AVERAGING) {
					if(total_exists > 0) {
						// now average the pixels and then assign it to correct point
						finalColor(0) = static_cast<uchar>(finalColor(0) / total_exists);
						finalColor(1) = static_cast<uchar>(finalColor(1) / total_exists);
						finalColor(2) = static_cast<uchar>(finalColor(2) / total_exists);
					}
				}
				
				uchar r = static_cast<uchar>(finalColor(0));
				uchar g = static_cast<uchar>(finalColor(1));
				uchar b = static_cast<uchar>(finalColor(2));
				result(y, x) = Vec3b(b, g, r);
			}
		}

		finalImage = result;
		namedWindow("result");
		imshow("result", finalImage);
		waitKey(0);
		imwrite(this->resourcePath + this->file.outputFileName, finalImage);
	} 
}

double Application::pointToLineDistance(Point2d& startPoint, Point2d& endPoint, Point2d& targetPoint) {

	// calculate vector
	Vec2d line(endPoint.x - startPoint.x, endPoint.y - startPoint.y);

	double u = ((targetPoint.x - startPoint.x)*(targetPoint.x - endPoint.x) + (targetPoint.y - startPoint.y)*(targetPoint.y - endPoint.y))/line.dot(line);

	Point2d xPoint(startPoint.x + u*(endPoint.x - startPoint.x),
				   startPoint.y + u*(endPoint.y - startPoint.y));

	// lenght of vector of xPoint-targetPoint is distance
	Vec2d targetLine(targetPoint.x - xPoint.x, targetPoint.y - xPoint.y);
	double distance = sqrt(targetLine.ddot(targetLine));
	return distance;
}

// this method returns the min distance to any edge within the rectange
// not which edge
double Application::pointToRectEdges(Mat& corners, Point2d& point) {
	
	Point2d corner1(corners.at<double>(0, 0), corners.at<double>(1, 0));
	Point2d corner2(corners.at<double>(0, 1), corners.at<double>(1, 1));
	Point2d corner3(corners.at<double>(0, 2), corners.at<double>(1, 2));
	Point2d corner4(corners.at<double>(0, 3), corners.at<double>(1, 3));

	double distance1 = pointToLineDistance(corner1, corner2, point);
	double distance2 = pointToLineDistance(corner2, corner3, point);
	double distance3 = pointToLineDistance(corner3, corner4, point);
	double distance4 = pointToLineDistance(corner4, corner1, point);

	return std::min(distance1, min(distance2, min(distance3, distance4)));
}

// run method runs the application by using the inifile parameters
void Application::run() {

	if(!allOk) {
		cerr<<"\n ini file non existent/not proper";
		cerr<<"\n please correct mistakes and run";
		return;
	}
	// init all fields from the inifile
	this->setApplicationResourcePath(this->file.resource);
	this->setFileName(this->file.fileName);
	this->setFileType(this->file.fileType);
	this->setStartIndex(this->file.startIndex);
	this->setEndIndex(this->file.endIndex);
	this->N = (endIndex - startIndex) + 1;
	this->setCenterIndex(this->file.centerIndex);
	this->setQueueIndex(this->file.queueIndex);
	matchPointDir = string(".\\");			// current directory is default path

	// SEQUENCE of flow
	// first read input images
	vector<Mat> inputImages;
	vector<Mat> homographyMatrices;
	vector<vector<Point2f>> srcPoints;
	vector<vector<Point2f>> destPoints;

	this->readImages(inputImages);

	if(file.resize) {
		// check if resize required 
		resizeImages(inputImages, Size(file.resizeWidth, file.resizeHeight));
	}

	// read matching points and read matrices should be mutually exclusive
	// so if we have homography matrices first, then we can use them directly
	// else check for points and proceed
	if(file.readMatrices) {
		readHomographyMatrices(homographyMatrices, resourcePath);
	}

	if(!file.readMatrices && file.readPoints) {
		readMatchingPoints(srcPoints, destPoints, resourcePath);

		// then compute the homography matrices
		for(int i = 0; i < inputImages.size() - 1; i++) {
			//homographyMatrices.push_back(computeHomography(srcPoints.at(i), destPoints.at(i), file.homography));
			homographyMatrices.push_back(computeHomographyNormalized(srcPoints.at(i), destPoints.at(i), Point2i(800, 600), Point2i(800, 600), file.homography));
		}
	}

	if(!file.readMatrices && !file.readPoints) {
		// manual user input
		// call method
		this->matchPoints(inputImages, resourcePath, true, &homographyMatrices);
	}


	// finally stitch planar panaroma
	stitchPlanarPanaroma(inputImages, homographyMatrices, file.queueIndex, file.stitch);
}