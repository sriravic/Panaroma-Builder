#include "Application.h"

int main(int argc, char** argv)
{
	/*
	// init the application
	string yardResource = "E:\\Studies\\Summer 2012\\Computer Vision\\Assignments\\assmt1\\resources\\yard\\";
	string roboticsLabResource = "E:\\Studies\\Summer 2012\\Computer Vision\\Assignments\\assmt1\\resources\\";
	string shanghaiResource = "E:\\Studies\\Summer 2012\\Computer Vision\\Assignments\\assmt1\\resources\\myshanghai\\";
	string myRoomResource = "E:\\Studies\\Summer 2012\\Computer Vision\\Assignments\\assmt1\\resources\\myroom";
	Application app(shanghaiResource,"shanghai", ".png", 0, 4, 2);				// center index = image(index).jpg --> which image file???
	vector<Mat> inputImages;
	
	vector<vector<Point2f>> srcPoints;
	vector<vector<Point2f>> destPoints;
	vector<Mat> homographyMatrices;
	app.readImages(inputImages);
	Application::resizeImages(inputImages, Size(800, 600));
	
	
	app.readMatchingPoints(srcPoints, destPoints, shanghaiResource);
	
	for(int i = 0; i < inputImages.size() - 1; i++) {
		cout<<i;
		homographyMatrices.push_back(Application::computeHomography(srcPoints.at(i), destPoints.at(i), OPENCV));
		cout<<i;
	}
	
	//app.matchPoints(inputImages, shanghaiResource, true, &homographyMatrices);
	//app.readHomographyMatrices(homographyMatrices, roboticsLabResource);
	app.stitchPlanarPanaroma(inputImages, homographyMatrices, 2, MY_METHOD);			// center index(zero indexed) ---> 3rd image in sequence
	*/

	if(argc != 2) {
		cerr<<"\n Invalid number of arguments.";
		cerr<"\n usage: PanaromaBuilder iniFile";
		return -1;
	}

	Application app(argv[1]);
	app.run();
	return 0;

}