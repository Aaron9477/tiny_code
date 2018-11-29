#include "tinyxml.h"
#include <iostream>
#include <string>
#include <iomanip>
#include <sstream>
using namespace std;

string int_to_string(int input_int);

string int_to_6_string(int input_int);

enum SuccessEnum {FAILURE, SUCCESS};

SuccessEnum saveXML(string pic_name, string pic_path, string pic_type, int coor_xmin, 
                    int coor_ymin, int coor_xmax, int coor_ymax, string xml_dir);
