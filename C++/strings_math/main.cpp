#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include <numeric>
#include <sstream>
#include <algorithm>


int main()
{
    std::vector<std::string> strVec(10);

    std::string str1("I'm a string");
    strVec[0] = str1;

    std::cout << str1.front() << "" << str1.back() << "\n";
    std::cout << "Length " << str1.length() << "\n";


    std::string str2(str1);
    strVec[1] = str2;


    std::string str3("String TEST",4);//, 4); // everything after 4th character // DOESN'T WORK IF I USE STRING VARIABLE AS FIRST ARGUMENT

    
    strVec[2] = str3;
    

    
    std::string str4(5, 'x'); // make xxxxx
    strVec[3] = str4;


    strVec[4] = str1.append(" and you're not");
    str1 += " and you're not";
    std::cout << str1 << "\n";
    
    str1.append(str1, 34, 37); // starting at 34, copy characters at 35,36,37
    strVec[5] = str1;
    str1.erase(13, str1.length()-1);
    strVec[6] = str1;
    
    std::reverse(str1.begin(), str1.end());
    std::cout << "Reverse " << str1 << "\n";

    std::transform(str2.begin(), str2.end(), str2.begin(),
                    toupper);
    std::cout << "Upper " << str2 << "\n";

    std::transform(str2.begin(), str2.end(), str2.begin(),
                    tolower);
    std::cout << "lower " << str2 << "\n";

    // a-z : 97 - 122
    // A-Z : 65 - 90
    char mychar = 'A';
    int myCI = mychar;
    std::cout << "A Code " << (int)'a' << "\n";

    std::string strNum = std::to_string(1+2);
    std::cout << "String " << strNum << "\n";

    if(str1.find("string") != std::string::npos)
        std::cout << "1st index " << str1.find("string") << "\n";

    std::cout << "Substr " << str1.substr(6, 6) << "\n"; //start at index 6 and get 6 characters
     

    for(auto y : strVec)
    {
        std::cout << y << "\n";
    }
    

        // ----- PROBLEM : SECRET STRING -----
    // Receive an uppercase string and hide its meaning
    // by turning it into ascii codes
    // Translate it back to the original letters
    
    std::string normalStr, secretStr = "";
    std::cout << "Enter your string in uppercase : ";
    std::cin >> normalStr;
    
    // Cycle through each character converting 
    // them into ascii codes which are stored in
    // a string
    for(char c: normalStr)
        secretStr += std::to_string((int)c);
        // secretStr += std::to_string((int)c - 23);
    
    std::cout << "Secret : " << secretStr << "\n";
    
    normalStr = "";
    
    // Cycle through numbers in string 2 at a time
    for(int i = 0; i < secretStr.length(); i += 2){
        
        // Get the 2 digit ascii code
        std::string sCharCode = "";
        sCharCode += secretStr[i];
        sCharCode += secretStr[i+1];
        
        // Convert the string into int
        int nCharCode = std::stoi(sCharCode);
        
        // Convert the int into a char
        char chCharCode = nCharCode;
        // char chCharCode = nCharCode + 23;
        
        // Store the char in normalStr
        normalStr += chCharCode;
    }
    
    std::cout << "Original : " << normalStr << "\n";
    
    // ----- END OF PROBLEM : SECRET STRING -----
    
    // ----- BONUS PROBLEM -----
    // Allow the user to enter upper and lowercase
    // letters by subtracting and adding 1 value
    // ----- END OF BONUS PROBLEM -----





    return 0;
}