#include "user.h"


int User::get_user_count()
{
    return user_count;
}

std::string User::get_status()
{
    return status;
}

void User::set_status(std::string status)
{
    if(status == "Gold" || status == "Silver" || status == "Bronze"){
        this->status = status;
    }
    else{
        this->status = "No status";
    }
}

User::User()
{
    //std::cout << "User created" << std::endl;
    user_count++;
}
User::User(std::string fn, std::string ln, std::string status)
{
    this->first_name = fn;
    this->last_name = ln;
    this->status = status;
    user_count++;
}

User::~User()
{
    // std::cout << "Destructor\n";
    user_count--; 
}
void User::output()
{
    //std::cout << "I am a user\n";
}

std::ostream& operator << (std::ostream& output, const User user);
std::istream& operator >> (std::istream &input, User  &user);




int User::user_count = 0;

void output_status(User user)
{
    std::cout << user.status;
}

int add_user_if_not_exists(std::vector<User> &users, User user)
{
    for(int i =0; i< users.size();i++)
    {
        if(users[i].first_name == user.first_name && users[i].last_name == user.last_name)
        {
            return i;
        }
    }
    users.push_back(user);
    return users.size() - 1;
}


std::ostream& operator << (std::ostream& output, const User user)
{
    output << "First name: " << user.first_name << "\nLast name: " << user.last_name << "\nStatus: " << user.status;
    return output;
}

std::istream& operator >> (std::istream &input, User  &user)
{
    input >> user.first_name >> user.last_name >> user.status;
    return input;
}