#ifndef USER_H
#define USER_H

#include <string>
#include <iostream>
#include <vector>

class User
{
    static int user_count;
    std::string status = "Gold";
    

    public:
        static int get_user_count();

        std::string first_name;
        std::string last_name;

        std::string get_status();

        void set_status(std::string status);

        User();
        User(std::string fn, std::string ln, std::string status);

        ~User();
        virtual void output(); // Because virtual is here, then if there is a subclass that also has output, it will be that output that is called, not User's. Polymorphism! yay
        friend void output_status(User user);
        friend std::ostream& operator << (std::ostream& output, const User user);
        friend std::istream& operator >> (std::istream &input, User  &user);

};
#endif