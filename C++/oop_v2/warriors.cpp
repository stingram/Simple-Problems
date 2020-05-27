#include <vector>
#include <sstream>
#include <iostream>
#include <math.h>
#include <numeric>
#include <algorithm>
#include <ctime>


class Warrior{

    private:
        int attack_max;
        int block_max;

    public:
        std::string name;
        int health;

        Warrior(std::string name, int health,
                    int attack_max, int block_max)
        {
            this->name = name;
            this->health = health;
            this->attack_max = attack_max;
            this->block_max = block_max;
        }

        int attack()
        {
            return std::rand() % this->attack_max;
        }
        int block()
        {
            return std::rand() % this->block_max;
        }

};


class Battle
{
    public:
        static void start_fight(Warrior& war1, Warrior& war2)
        {
            while(true)
            {
                if(Battle::get_attack_result(war1, war2).compare("Game Over") == 0)
                {
                    std::cout << "Game Over!\n";
                    break;
                }
                if(Battle::get_attack_result(war2, war1).compare("Game Over") == 0)
                {
                    std::cout << "Game Over!\n";
                    break;
                }
            }
        }

        static std::string get_attack_result(Warrior& war1, Warrior& war2)
        {
            int attack_1 = war1.attack();
            int block_2 = war2.block();
            int damage = ceil(attack_1-block_2);
            damage = (damage <= 0) ? 0 : damage;
            war2.health = war2.health - damage;

            std::cout << war1.name << " attacks " << war2.name <<
                " and deals " << damage << " damage.\n";
            std::cout << war2.name << " is down to " << war2.health <<
                " health.\n";

            if(war2.health <= 0)
            {
                std::cout << war2.name << " has died and " << 
                    war1.name << " is victorious.\n";
                return "Game Over";
            }
            else
            {
                return "Fight Again!";
            }
            
        }

};

int main()
{
    srand(time(NULL));
    Warrior thor("Thor", 100, 30, 15);
    Warrior hulk("Hulk", 135, 25, 10);

    Battle::start_fight(thor, hulk);


    return 0;
}