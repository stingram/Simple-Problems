#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include <ctime>
#include <numeric>
#include <cmath>
#include <sstream>
#include <iterator>
#include <memory>
#include <stdio.h>
 
// A Smart pointer is a class that provides the 
// power of pointers, but also handles the reallocation
// of memory when it is no longer required (The pointer
// is automatically destroyed)
 
// typedef creates an alias for a more complex type name
typedef std::vector<int32_t> intVec;

// Here I demonstrate how to use templates 
// polymorphically 
 
// Base class all pizzas inherit along with MakePizza
// which will be overridden
class Pizza
{
    public:
    virtual void make_pizza() = 0;

};

// The last templates that will be called
class NY_style_crust
{
    public:
        std::string add_ingredients()
        {
            return "Crust so thin you can see through it \n\n";
        }
};

class Deepdish_crust
{
    public:
        std::string add_ingredients()
        {
            return "Super awesome chicago deep dish\n\n";
        }
};
// End of last templates called

// The middle templates called
template<typename T>
class lots_of_meat : public T
{
    public: std::string add_ingredients()
    {
        return "Lots of random meat, " + T::add_ingredients();
    }
};

template<typename T>
class vegan : public T
{
    public: std::string add_ingredients()
    {
        return "Vegan cheese, veggies, " + T::add_ingredients();
    }
};
// End of middle templates called

// We inherit from Pizza as well as the initial next template
template <typename T>
class meat_ny_style: public T, public Pizza
{
    public:
        void make_pizza()
        {
            std::cout << "Meat NY Style Pizza : " << 
                T::add_ingredients();
        }
};

template <typename T>
class vegan_deep_dish: public T, public Pizza
{
    public:
        void make_pizza()
        {
            std::cout << "Vegan Deep Dish : " << 
                T::add_ingredients();
        }
};


int main()
{
    // When you define a primitive type like int or
    // float you define exactly the amount of space
    // to set aside
    
    // If you need to define how much space to set aside
    // you could call malloc() and tell it how much
    // space to set aside and it returns the address to
    // that memory address
    /*
    int amt_to_store;
    std::cout << "How many numbers do you want to store: ";
    std::cin >> amt_to_store;
    int* nums;
    nums = (int *) malloc(amt_to_store * sizeof(int));
    if(nums != NULL)
    {
        int i =0;
        while(i < amt_to_store)
        {
            std::cout << "Enter a number : ";
            std::cin >> nums[i];
            i++;
        }
    }
    std::cout << "You entered these numbers\n";
    for(int i=0; i < amt_to_store; i++)
    {
        std::cout << nums[i] << "\n";
    }

    delete nums;
    */

   // Smart pointer way
    int amt_to_store;
    std::cout << "How many numbers do you want to store: ";
    std::cin >> amt_to_store;

    std::unique_ptr<int[]> nums(new int[amt_to_store]);

    // NOT ALLOWED
    // std::unique_ptr<int[]> num2 = nums; 

    if(nums != NULL)
    {
        int i =0;
        while(i < amt_to_store)
        {
            std::cout << "Enter a number : ";
            std::cin >> nums[i];
            i++;
        }
    }
    std::cout << "You entered these numbers\n";
    for(int i=0; i < amt_to_store; i++)
    {
        std::cout << nums[i] << "\n";
    }

    // unique_ptr is a smart pointer that disposes of
    // a pointer when it is no longer in use
    std::vector<std::unique_ptr<Pizza>> pizza_orders;

    // Generate Pizza types and place them at the end of the vector
    pizza_orders.emplace_back(new meat_ny_style<lots_of_meat<NY_style_crust>>());
    pizza_orders.emplace_back(new vegan_deep_dish<vegan<Deepdish_crust>>());

    // Call the pizzas and execute the directions 
    // for making them
    for (auto& pizza: pizza_orders)
    {
        pizza->make_pizza();
    }

    return 0;
}