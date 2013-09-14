#include "Application.h"

// initialize static members
string IniFile::iniStrings[] = {"[RESOURCE]", "[FILENAME]", "[FILETYPE]", "[RESIZE]", "[RESIZE_WIDTH]", "[RESIZE_HEIGHT]", 
	"[START_INDEX]", "[END_INDEX]", "[CENTER_INDEX]", "[QUEUE_INDEX]", "[READ_POINTS]", "[READ_MATRICES]", "[HOMOGRAPHY]", "[STITCH]", "[BLEND]", "[OUTPUT]"};



string IniFile::getValue(string iniString) {
	size_t equals;
	equals = iniString.find("=");
	if(iniString.at(equals + 1) == ' ') 
		equals = equals + 2;
	// now copy from equals till end
	string retString = iniString.substr(equals);
	return retString;
}

bool IniFile::parseIniFile(string file) {

	ifstream input(file);
		if(input.good()) {
			// read off values
			// see ini file structure 
			string temp;
			char buffer[1024];
			
			while(!input.eof()) {
				string inputString;
				//input>>inputString;
				input.getline(buffer, 1024);
				inputString = string(buffer);
				size_t position = inputString.find("#");
				if(position == string::npos && inputString.length() > 1) {
					// not a comment
					// safely ignore all other lines

					// check if there is keyword
					size_t found;
					
					
					for(int i = 0; i < NUM_ATTRIBUTES; i++) {
						found = inputString.find(IniFile::iniStrings[i]);
						if(found != string::npos) {
							switch(i) {
							case 0:
								this->resource = getValue(inputString);
								break;
							case 1:
								this->fileName = getValue(inputString);
								break;
							case 2:
								this->fileType = getValue(inputString);
								break;
							case 3:
								temp = getValue(inputString);
								if(temp == "yes" || temp == "YES") this->resize = true;
								else this->resize = false;
							case 4:
								temp = getValue(inputString);
								// convert to integer
								this->resizeWidth = atoi(temp.c_str());
								break;
							case 5:
								temp = getValue(inputString);
								this->resizeHeight = atoi(temp.c_str());
								break;
							case 6:
								temp = getValue(inputString);
								this->startIndex = atoi(temp.c_str());
								break;
							case 7:
								temp = getValue(inputString);
								this->endIndex = atoi(temp.c_str());
								break;
							case 8:
								temp = getValue(inputString);
								this->centerIndex = atoi(temp.c_str());
								break;
							case 9:
								temp = getValue(inputString);
								this->queueIndex = atoi(temp.c_str());
								break;
							case 10:
								temp = getValue(inputString);
								if(temp == "yes" || temp == "YES") this->readPoints = true;
								else this->readPoints = false;
								break;
							case 11:
								temp = getValue(inputString);
								if(temp == "yes" || temp == "YES") this->readMatrices = true;
								else this->readMatrices = false;
								break;
							case 12:
								temp = getValue(inputString);
								if(temp == "MY_METHOD") this->homography = MY_METHOD;
								else if(temp == "OPENCV") this->homography = OPENCV;
								break;
							case 13:
								temp = getValue(inputString);
								if(temp == "MY_METHOD") this->stitch = MY_METHOD;
								else if(temp == "OPENCV") this->stitch = OPENCV;
								break;
							case 14:
								temp = getValue(inputString);
								if(temp == "AVERAGING") this->blend = AVERAGING;
								else if(temp == "DISTANCE") this->blend = DISTANCE;
								break;
							case 15:
								this->outputFileName = getValue(inputString);
								break;
							};
						}
					}
					
				} else continue;
			}
		} else {
			cerr<<"\n Invalid ini file";
			return false;
		}

		return true;
}

