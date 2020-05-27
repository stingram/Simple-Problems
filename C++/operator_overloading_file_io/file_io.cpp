#include <cstdlib>
#include <iostream>
#include <sstream>
#include <vector>
#include <ctime>
#include <string>
#include <numeric>
#include <fstream>


 
std::vector<std::string> StringToVector(std::string, 
        char separator);


int main()
{
    std::ofstream write_to_file;
    std::ifstream read_from_file;
    std::string text_to_write = "";
    std::string text_from_file = "";
    // We open the file by providing a name and then either
    // ios::app : Append to the end of the file
    // ios::trunc : If the exists delete content
    // ios::in : Open file for reading
    // ios::out : Open file for writing
    // ios::ate : Open writing and move to the end of the file

    write_to_file.open("test.txt", std::ios_base::out |
                    std::ios_base::trunc);
    if(write_to_file.is_open())
    {
        write_to_file << "Beginning of file\n";
        std::cout << "Enter data to rite : ";
        getline(std::cin, text_to_write);
        write_to_file << text_to_write;
        write_to_file.close();
    }

    read_from_file.open("test.txt", std::ios_base::in);
    if(read_from_file.is_open())
    {
        while(read_from_file.good())
        {
            getline(read_from_file, text_from_file);
            std::cout << text_from_file << "\n";


            // ----- PROBLEM -------------
            std::vector<std::string> vect = StringToVector(text_from_file, ' ');
            int words_in_line = vect.size();
            std::cout << "Words in line : " <<
                        words_in_line << "\n";
            int char_count = 0;
            for(auto word: vect)
            {
                for(auto letter: word)
                {
                    char_count++;
                }
            }
            int avg = char_count/words_in_line;
            std::cout << "Average word length : " <<
                    avg << "\n";
        }
        read_from_file.close();
    }

    return 0;
}


std::vector<std::string> StringToVector(std::string theString, 
        char separator){
 
    // Create a vector
    std::vector<std::string> vecsWords;
    
    // A stringstream object receives strings separated
    // by a space and then spits them out 1 by 1
    std::stringstream ss(theString);
    
    // Will temporarily hold each word in the string
    std::string sIndivStr;
    
    // While there are more words to extract keep
    // executing
    // getline takes strings from a stream of words stored
    // in the stream and each time it finds a blanks space
    // it stores the word proceeding the space in sIndivStr
    while(getline(ss, sIndivStr, separator)){
        
        // Put the string into a vector
        vecsWords.push_back(sIndivStr);
    }
    
    return vecsWords;
}