#include <iostream>
#include <fstream>
#include<time.h>

using namespace std;

int main()
{
    srand(time(0));
    int x;

    std::ofstream myfile;
    myfile.open ("data.csv");

    int t = 11;
    while(t--)
    {
        for(int i=0;i<150;i++)
        {
            x = rand()%20;
            myfile <<x <<"," <<endl;    
        }
        for(int i=0;i<15;i++)
        {
            x = ((rand()%30)+20);
            myfile <<x <<"," <<endl;    
        }
        for(int i=0;i<35;i++)
        {
            x = ((rand()%100)+50);
            myfile <<x <<"," <<endl;    
        }
        for(int i=0;i<15;i++)
        {
            x = ((rand()%30)+20);
            myfile <<x <<"," <<endl;    
        }
        for(int i=0;i<150;i++)
        {
            x = rand()%20;
            myfile <<x <<"," <<endl;    
        }
    }

    myfile.close();
    return 0;
}