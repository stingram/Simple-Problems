#include <iostream>
#include <fstream>
#include <vector>
#include <string>

int main()
{

    std::ifstream file ("hello2.txt"); // WILL NOT CREATE THE FILE IF IT DOESN'T EXIST

    char temp = file.get(); // gets character
    std::cout << temp << "\n";

    std::string line;
    getline(file, line);
    std::cout << line << "\n";


    /*
    std::vector<std::string> names;

    std::string input;
    while(file >> input) // returns file is successful
    {
        names.push_back(input);
    }

    for(std::string name : names)
    {
        std::cout << name << std::endl;
    }
    */


    /*
    std::ofstream file;
    file.open("hello.txt"); // this creates the file if it doesn't exist

    std::string filename;
    std::cin >> filename;

    // or
    //std::ofstream file2("hello2.txt"); // this doesn't actually open anything
    //std::ofstream file2 ("hello2.txt"); // THIS DOES OPEN AND CREATES THE FILE IF IT DOESN'T EXIST
    std::ofstream file2 (filename.c_str(), std::ios::app); // For appending, .c_str() is optional is c++11




    if(file2.is_open())
    {
        std::cout << "File opened." << std::endl;
    }

    std::vector<std::string> names;
    names.push_back("Caleb");
    names.push_back("Sally");
    names.push_back("Mike");

    for(std::string name : names){
        file2 << name << std::endl;
    }

    file.close();
    file2.close();
    */
    return 0;
}