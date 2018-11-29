#include <iostream>
#include <string>
#include <iomanip>
#include <sstream>
using namespace std;
int main(int argc, char const *argv[])
{
	ostringstream ss;

    int num = 100;
    int a=1;
    ss<<setfill('0')<<setw(6)<<a<<endl;
    cout << ss.str() <<endl;
    cout<< ss.str().c_str() <<endl;

    return 0;
}

