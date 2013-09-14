#include "Application.h"

// Source file for reading and writing matching points from file
// File extension is mpt
MatchingPoints::MatchingPoints() {
	directory = string(".\\");
}

MatchingPoints::MatchingPoints(string filedir, string srcImage, string destImage, vector<Point2f> src, vector<Point2f> dest) {
	
	directory = filedir;
	sourceImage = srcImage;
	destinationImage = destImage;
	srcPoints = src;
	destPoints = dest;
}

// misc functions
// reading and writing

void MatchingPoints::writeToFile() {

	string filename = directory + string("\\") + sourceImage + destinationImage + string(".mpt");
	ofstream opfile(filename);
	opfile<<sourceImage<<endl;
	opfile<<destinationImage<<endl;
	opfile<<srcPoints.size()<<endl;
	// we gotta write x,y as it is
	for(unsigned int i = 0; i < srcPoints.size(); i++) {
		opfile<<srcPoints.at(i).x<<" "<<srcPoints.at(i).y<<endl;
	}
	opfile<<destPoints.size()<<endl;
	for(unsigned int i = 0; i < destPoints.size(); i++) {
		opfile<<destPoints.at(i).x<<" "<<destPoints.at(i).y<<endl;
	}
	opfile.close();
}

void MatchingPoints::readFile(string filename, string dir) {

	// check if directory has last two characters as \\
	// else we have to append them to create a proper directory path
	if(dir.length() != 0) {
		directory = dir;
	}
	if(dir[dir.length()-1] != '\\') {
		directory = dir + string("\\");
	}
	ifstream ipfile(directory + filename);
	if(!ipfile.bad()) {
		string srcString;
		string destString;
		int srcSize, destSize;
		ipfile>>srcString;
		ipfile>>destString;
		ipfile>>srcSize;
		for(int i = 0; i < srcSize; i++) {
			int x, y;
			ipfile>>x>>y;
			srcPoints.push_back(Point2f(static_cast<float>(x), static_cast<float>(y)));
		}
		ipfile>>destSize;
		for(int i = 0; i < destSize; i++) {
			int x, y;
			ipfile>>x>>y;
			destPoints.push_back(Point2f(static_cast<float>(x), static_cast<float>(y)));
		}
	}
	ipfile.close();
}