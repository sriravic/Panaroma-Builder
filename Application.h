#ifndef __APPLICATION_H__
#define __APPLICATION_H__

#include "cv.h"
#include "highgui.h"
#include <vector>
#include <sstream>
#include <iostream>
#include <fstream>
using namespace std;
using namespace cv;


// all misc classes


// MatchingPoints class
// This is used to create objects that can be used to store and retrieve pair wise matches between images
// The file structure is as follows

/*
FILE STRUCTURE:
1. SrcImageName
2. DestImageName
3. Number of points in SrcImage selected
4. Number of points in DestImage selected
*/

class MatchingPoints
{
private:
	vector<Point2f> srcPoints;
	vector<Point2f> destPoints;
	string sourceImage;
	string destinationImage;
	string directory;				// directory where to save the file
public:
	MatchingPoints();
	MatchingPoints(string dir, string srcImage, string destImage, vector<Point2f> src, vector<Point2f> dest);
	// write to file
	void writeToFile(); 
	void readFile(string filename, string directory = ".\\");


	// get methods
	vector<Point2f> getSrcPoints() { return srcPoints; }
	vector<Point2f> getDestPoints() { return destPoints; }
	string getSourceImage() { return sourceImage; };
	string getDestinationImage() { return destinationImage; }
	string getDirectory() { return directory; }

};

// HomographyFile class
// This class will be used to store and retrieve homography matrices that have been calculated for images
class HomographyFile
{
private:
	string sourceImage;
	string destinationImage;
	string directory;
	Mat homographyMatrix;
public:
	HomographyFile();
	HomographyFile(string dir, string srcImage, string destImage, Mat h);
	void writeFile();
	void readFile(string filename, string dir = ".\\");
	
	// get methods
	Mat getHomographyMatrix() { return homographyMatrix; }
	string getSourceImage() { return sourceImage; }
	string getDestinationImage() { return destinationImage; }
	string getDirectory() { return directory; }
};

// Mouse Message structure.
typedef struct Message {
	char* window_name;
	vector<Point2f>* matchingPoints;
	Mat& image;
	Message(Mat& input, char* wname, vector<Point2f>* mpoints):image(input),matchingPoints(mpoints) {
		window_name = wname;
		//matchingPoints = mpoints;
	}
	Message(Message& M):image(M.image), matchingPoints(M.matchingPoints), window_name(M.window_name){}
}Message;

enum Method { MY_METHOD, OPENCV };
enum Panaroma { PLANAR, CYLINDRICAL };
enum Blend { AVERAGING, DISTANCE };


static const int NUM_ATTRIBUTES = 16;
struct IniFile {
	
	static string iniStrings[NUM_ATTRIBUTES];
	string resource;
	string fileName;
	string fileType;
	bool resize;
	int resizeWidth;
	int resizeHeight;
	int startIndex;
	int endIndex;
	int centerIndex;
	int queueIndex;
	bool readPoints;
	bool readMatrices;
	Method homography;
	Method stitch;
	Blend blend;
	string outputFileName;


	// ctor
	IniFile(){}
	// given an ini string, it splits the value and returns it back
	string getValue(string iniString); 
	// parse an input file and fill in the details
	bool parseIniFile(string file); 
};



class Application 
{
private:
	string iniFile;						// If using an inifile
	string resourcePath;				// location where images are stored
	string fileType;					// images file type eg: JPEG, PNG, BMP etc
	string fileName;					// The starting name of the files. All files should be named in sequence so that stitcher can work accordingly
	bool allOk;
	int startIndex;
	int endIndex;
	int N;								// Number of images to be considered for panaroma
	int centerIndex;					// the central image or target image for planar homography
	int queueIndex;						// center index in the queue of homography matrices. ----------> This was initially used in stitch method as the penultimate parameter
	string matchPointDir;				// if we have written the matchpoints already we can use this. We store the homography matrices along with match points only.. 
	Size resizeFactor;					// if homography was computed using a scale factor, we have to store that also so that when images are read back we have to scale them appropriately.
	IniFile file;

	

public:

	// Non static methods
	Application() {}
	Application(string inifile) {		// read from ini file.
		iniFile = inifile;
		allOk = file.parseIniFile(iniFile);
	}
	Application(string resPath, string fname, string type, int start, int end, int center) {
		resourcePath = resPath;
		fileName = fname;
		fileType = type;
		N = (end - start) + 1;					// zero indexing
		startIndex = start;						// there might be times where we just want a segment of input images among the n
		endIndex = end;
		centerIndex = center;
		matchPointDir = string(".\\");			// current directory is default path
	}
	void run();

	// setter methods
	void setApplicationResourcePath(string path) { resourcePath = path; }
	void setFileName(string name) { fileName = name; }
	void setFileType(string type) { fileType = type; }
	void setImageCount(int n) { N = n; }
	void setCenterIndex(int n) { centerIndex = n; }
	void setMatchPointDir(string dir) { matchPointDir = dir; }
	void setStartIndex(int i) { startIndex = i; }
	void setEndIndex(int i) { endIndex = i; }
	void setQueueIndex(int i ) { queueIndex = i; }


	// getter methods
	string getApplicationResourcePath() { return resourcePath; }
	string getFileName() { return fileName; }
	string getFileType() { return fileType; }
	string getMatchPointDir() { return matchPointDir; }
	int getImageCount() { return N; }
	int getCenterIndex() { return centerIndex; }
	int getStartIndex() { return startIndex; }
	int getEndIndex() { return endIndex; }
	int getQueueIndex() { return queueIndex; }



	// Misc method
	// The Application has to be initialized with all required values before any images can be read.
	// Then we can provide one input array that will read off the images from the paths
	void readImages(vector<Mat>& inputImages);

	// ApplicationMatchPoints logic
	void matchPoints(vector<Mat>& inputImages, string outputDir, bool computeHomographies, vector<Mat>* homographyMatrices);
	// this method reads a matching point file given the source and destination index
	//bool readMatchingPoints(vector<Point2f> srcPoints, vector<Point2f> destPoints, int srcIndex, int destIndex);
	// this method reads off all the matching points for the application initialized images
	void readMatchingPoints(vector<vector<Point2f>>& srcPoints, vector<vector<Point2f>>& destPoints, string dir);
	void readHomographyMatrices(vector<Mat>& outputHomographies, string dir);

	// useful static methods
	static Scalar Bilerp(Mat& src, Point2d point);
	// given an image and a transformation matrix, this method will calculate the bounding box of the new image
	static Rect calculateBoundingBox(Mat& image, Mat& H);	

	//This method will be used to get the 4 transformed corners of the transformed image so that we can accurately get a sense of where pixels go.
	// same as calculateBounding box but only returning coords are different
	// The returned matrix will have coords in 2x4 matrix, column wise indicating points
	static Mat calculateXfmImageCorners(Mat& image, Mat& H);

	// Get the final homography matrix from images index towards the central image
	static Mat getFinalHomographyMatrix(vector<Mat>& H, int center_index, int src_index);
	// one time calculation for all homography matrices in an image sequence
	static vector<Mat> calculateAllFinalHomographyMatrices(vector<Mat>& H, int center_index);
	// convert input image to cylindrical coords
	static Mat getCylindricalProjection(Mat& input, double focal);					
	static void onMouseMessage(int event, int x, int y, int flags, void* msg);
	
	// given a point we can find the distance of the point to line joining start point and end point
	static double pointToLineDistance(Point2d& startPoint, Point2d& endPoint, Point2d& targetPoint);
	static double pointToRectEdges(Mat& corners, Point2d& point);

	// this method is implementation of my own compute homography and method
	static Mat computeHomography(vector<Point2f>& src, vector<Point2f>& dest, Method m = MY_METHOD);
	static Mat computeHomographyNormalized(vector<Point2f>& src, vector<Point2f>& dest, Point2i srcDim, Point2i destDim, Method m = MY_METHOD);
	void stitchPlanarPanaroma(vector<Mat>& images, vector<Mat>& homographies, int center_index, Method m = MY_METHOD);
	// cylindrical panaroma
	static void stitchCylindricalPanaroma(vector<Mat>& images, vector<Mat>& homographies);			
	
	// resize all images
	static void resizeImages(vector<Mat>& inputImages, Size sz) {
		for(int i = 0; i < inputImages.size(); i++) {
			cv::resize(inputImages.at(i), inputImages.at(i), sz);
		}
	}

	static inline bool isWithinBounds(Point2d& point, Mat& image) {
		// this method returns true if required point is within image bounds
		if(point.x >= 0 && point.x < static_cast<double>(image.cols) && point.y >= 0 && point.y < static_cast<double>(image.rows)) return true; 
		else return false;
	}



};

struct DistancePoint
{
	int imageIndex;
	double distance;
	DistancePoint() { imageIndex = -1; distance = 0; }
	DistancePoint(int index, double d) { imageIndex = index; distance = d; }
};




#endif