#include "Application.h"

// HomographyFile class definition

// ctor
HomographyFile::HomographyFile() {
	directory = string(".\\");		// current directory is default unless overriden by the user
}

HomographyFile::HomographyFile(string dir, string srcImage, string destImage, Mat h) {
	directory = dir;
	sourceImage = srcImage;
	destinationImage = destImage;
	homographyMatrix = h;
}

// misc methods
// read and write
void HomographyFile::writeFile() {
	
	ofstream outputfile(directory + string("\\") + sourceImage + destinationImage + ".hmg", ios::binary);
	outputfile<<sourceImage<<endl;
	outputfile<<destinationImage<<endl;
	for(int row = 0; row < homographyMatrix.rows; row++) {
		for(int col = 0; col < homographyMatrix.cols; col++) {
			stringstream dbl;
			dbl.precision(std::numeric_limits<double>::digits10);
			dbl<<homographyMatrix.at<double>(row, col);
			outputfile<<dbl.str()<<" ";
		}
		outputfile<<endl;
	}
}

// write methods
void HomographyFile::readFile(string file, string dir) {
	
	if(dir.length() != 0) {
		directory = dir;
	}
	if(dir[dir.length()-1] != '\\') {
		directory = dir + string("\\");
	}
	ifstream inputfile(directory + file, ios::binary);
	inputfile>>sourceImage;
	inputfile>>destinationImage;
	homographyMatrix = Mat(3, 3, CV_64F);
	for(int row = 0; row < 3; row++) {
		for(int col = 0; col < 3; col++) {
			inputfile>>homographyMatrix.at<double>(row, col);
		}
	}


}