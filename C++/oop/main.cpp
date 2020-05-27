#include <iostream>
#include <string>
#include "user.h"
#include "teacher.h"
#include "student.h"


void do_something(User& user)
{
    user.output();
}

int main()
{

    /*
    User me;
    me.first_name = "Caleb";
    me.last_name = "Curry";
    

    std::cout << "Status: " << me.get_status() << std::endl;

    std::vector<User> users;
    users.push_back(me);


    User user2;
    user2.first_name = "S";
    user2.last_name = "I";

    User user3;
    user3.first_name = "SS";
    user3.last_name = "II";

    users.push_back(user2);
    users.push_back(user3);

    User user4;
    user4.first_name = "Ss";
    user4.last_name = "I";

    std::cout << add_user_if_not_exists(users, user4) << std::endl;
    std::cout << users.size() << std::endl;
    */

    // User user5("Steven", "Ingram", "Silver");
    // std::cout << user5.get_status() << std::endl;

    /*
    User user,user1, user2;
    user.set_status("Tacos");
    std::cout << User::get_user_count() << std::endl;
    */
   // User user;
   //user.first_name ="S";
   //user.last_name = "I";
   /*
   user.set_status("Gold");

   std::cin >> user;
   std::cout << user << std::endl;
    */
   // output_status(user);

    /*
    user.first_name ="S";
    user.last_name = "I";
    user.set_status("Gold");
    */
    /*
    std::cin >> user;
    std::cout << user << std::endl;
    */
    Teacher teacher;
    Student student;
    User& u1 = teacher;
    User& u2 = student; // Making a reference to a user
    do_something(u1);
    do_something(u2);

    /*
    teacher.first_name = "Teach";
    std::cout << teacher.first_name << std::endl;
    teacher.output();
    */

    return 0;
}