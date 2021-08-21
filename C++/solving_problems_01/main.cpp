#include <cstdlib>
#include <iostream>
#include <vector>
#include <numeric>
#include <sstream>
#include <ctime>

std::vector<std::string> string_to_vector(const std::string in_str, const char sep);
std::string vector_to_string(const std::vector<std::string> in_vec);
std::string trim_white_space(std::string);
std::vector<int> find_substring_matches(const std::string in_str, const std::string query);
std::string replace_all_substrings(std::string in_str, const std::string old_substring, const std::string new_substring);
std::string caesar_cipher(std::string in_str, const int shift, bool enc_flag);
void solve_for_x(std::string equation);
std::vector<int> Range(int start, int max, int step);

bool is_prime(const int num);
std::vector<int> get_primes(const int max_num);
std::vector<int> generate_random_vector(int amount, int min, int max);


int main()
{
   // ----- 6. CHARACTER FUNCTIONS -----
    char letterZ = 'z';
    char num3 = '3';
    char aSpace = ' ';
    
    std::cout << "Is z a letter or number " << 
            isalnum(letterZ) << "\n";
    std::cout << "Is z a letter " << 
            isalpha(letterZ) << "\n";
    std::cout << "Is z uppercase " << 
            isupper(letterZ) << "\n";
    std::cout << "Is z lowercase " << 
            islower(letterZ) << "\n";
    std::cout << "Is 3 a number " << 
            isdigit(num3) << "\n";
    std::cout << "Is space a space " << 
            isspace(aSpace) << "\n";
    
    // ----- END CHARACTER FUNCTIONS -----


    std::string theString = "Make me secret";
    
    std::string encryptedStr = caesar_cipher(theString, 
            5, true);
    
    std::string decryptedString = caesar_cipher(encryptedStr, 
            5, false);
    
    std::cout << "Encrypted " << encryptedStr << "\n";
    
    std::cout << "Decrypted " << decryptedString << "\n";



    // ---------    SOLVE FOR X ----------------
    std::cout << "Enter equ to solve: ";
    std::string equation = "";
    getline(std::cin, equation);
    solve_for_x(equation); 


    // ------------ IS PRIME ---------------
    int num = 0;
    std::cout << "Number to check: ";
    std::cin >> num;

    std::cout.setf(std::ios::boolalpha); // outputs true or false instead of 1 or 0
    std::cout << "Is " << num << " Prime " << is_prime(num) << "\n";
    int max_prime;
    std::cin >> max_prime;
    std::vector<int> prime_list = get_primes(max_prime);
    for(auto x: prime_list)
    {
        std::cout << x << "\n";
    }

    // -----------   RANDOM NUMBER GENERATOR ---------
    std::vector<int> vals = generate_random_vector(10, 5, 50);
    for(auto x: vals)
    {
        std::cout << x << "\n";
    }


}

std::vector<std::string> string_to_vector(const std::string in_str, const char sep)
{
    std::vector<std::string> words;
    std::stringstream ss(in_str);

    std::string temp;

    while(getline(ss, temp, sep))
    {
        words.push_back(temp);
    }
    return words;


}

std::string vector_to_string(const std::vector<std::string> in_vec)
{
    std::string out_string ="";
    for(std::string s : in_vec)
    {
        out_string += s + " ";
    }
    out_string += "\n";
    return out_string;
}

std::string trim_white_space(std::string in_string)
{
    std::string whitespaces(" \n\r\t\f");
    in_string.erase(in_string.find_last_not_of(whitespaces)+1); // need +1 because we are getting the index of the thing that ISN'T whitespace
    in_string.erase(0, in_string.find_first_not_of(whitespaces)); // don't need +1 because convention is [start,end)
    return in_string;
}

std::vector<int> find_substring_matches(const std::string in_str, const std::string query)
{
    std::vector<int> out_vec;
    int index = in_str.find(query, 0);
    while(index != std::string::npos)
    {
        out_vec.push_back(index);
        index = in_str.find(query, index+1);
    }
    return out_vec;
}

std::string replace_all_substrings(std::string in_str, const std::string old_substring, const std::string new_substring)
{
    std::vector<int> matches = find_substring_matches(in_str, old_substring);

    if(matches.size() > 0)
    {
        int length_diff = new_substring.size() - old_substring.size();
        int loop_count = 0;
        for (int index : matches)
        {
            in_str.replace(index + (length_diff*loop_count),
                           new_substring.size(),
                           new_substring);
            loop_count++;
        }

    }
    return in_str;

}


std::string caesar_cipher(std::string in_str, int shift, bool enc_flag)
{
    std::string out_str = "";
    int charCode = 0;
    char letter;

    if(enc_flag)
    {
        shift *= -1;
    }
    for(char& c: in_str)
    {
        if(isalpha(c))
        {
            charCode = (int)c;
            charCode += shift;

            if(isupper(c))
            {
                if(charCode >(int)'Z')
                {
                    charCode -= 26;
                } else if(charCode < (int)'A')
                {
                    charCode +=26;
                }
            }
            else
            {
                if(charCode >(int)'z')
                {
                    charCode -= 26;
                } else if(charCode < (int)'a')
                {
                    charCode +=26;
                }
            }
            letter = charCode; // from int to string
            out_str += letter;

        }
        else{
            letter = c;     // don't change non-alpha character
            out_str += letter; 
        }
    }
    return out_str;
}


// x + 4 = 9
void solve_for_x(std::string equation)
{
    std::vector<std::string> vec_equation = string_to_vector(equation, ' ');
    int num1 = std::stoi(vec_equation[2]);
    int num2 = std::stoi(vec_equation[4]);
    int xVal = num2 - num1;
    std::cout << "x = " << xVal << "\n";
}

std::vector<int> Range(int start, int max, int step)
{
    
    // Every while statement needs an index 
    // to start with
    int i = start;
    
    // Will hold returning vector
    std::vector<int> range;
    
    // Make sure we don't go past max value
    while(i <= max){
        
        // Add value to the vector
        range.push_back(i);
        
        // Increment the required amount
        i += step;
    }
    
    return range;
}    


bool is_prime(const int num)
{
    for(auto n : Range(2, ceil(num/2) - 1, 1))
    {
        if((num % n) == 0)
            return false;
    }
    return true;
}

std::vector<int> get_primes(const int max_num)
{
    std::vector<int> primes;
    for(auto x: Range(2, max_num, 1))
    {
        if(is_prime(x))
        {
            primes.push_back(x);
        }
    }
    return primes;
}

std::vector<int> generate_random_vector(int amount, int min, int max)
{
    std::vector<int> vals;
    srand(time(NULL)); // make seed
    int i =0, rand_val = 0;
    while(i < amount)
    {
        rand_val = min + std::rand() % ((max + 1) - min); // guarantee it falls in requested range
        vals.push_back(rand_val);
        i++;
    }

    return vals;
}