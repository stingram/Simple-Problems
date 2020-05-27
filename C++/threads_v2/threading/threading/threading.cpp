// threading.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "pch.h"
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include <ctime>
#include <numeric>
#include <cmath>
#include <sstream>
#include <thread>
#include <chrono>
#include <ctime>
#include <mutex>

int get_random(int max)
{
	srand(time(NULL));
	return rand() % max;
}

void execute_thread(int id)
{
	// Get current time
	auto nowTime = std::chrono::system_clock::now();

	// Convert to a time we can output
	std::time_t sleepTime =
		std::chrono::system_clock::to_time_t(nowTime);

	// Convert to current time zone
	tm myLocalTime = *localtime(&sleepTime);

	// Print full time information
	std::cout << "Thread " << id <<
		" Sleep Time : " <<
		std::ctime(&sleepTime) << "\n";

	// Get separate pieces
	std::cout << "Month : " <<
		myLocalTime.tm_mon << "\n";
	std::cout << "Day : " <<
		myLocalTime.tm_mday << "\n";
	std::cout << "Year : " <<
		myLocalTime.tm_year + 1900 << "\n";
	std::cout << "Hours : " <<
		myLocalTime.tm_hour << "\n";
	std::cout << "Minutes : " <<
		myLocalTime.tm_min << "\n";
	std::cout << "Seconds : " <<
		myLocalTime.tm_sec << "\n\n";


	std::this_thread::sleep_for(std::chrono::seconds(get_random(3)));
	nowTime = std::chrono::system_clock::now();
	sleepTime = std::chrono::system_clock::to_time_t(nowTime);

	std::cout << "Thread " << id <<
		" Awake Time : " <<
		std::ctime(&sleepTime) << "\n";


}


std::string get_time()
{
	auto now_time = std::chrono::system_clock::now();
	std::time_t sleep_time = std::chrono::system_clock::to_time_t(now_time);
	return std::ctime(&sleep_time);
}

double balance = 100;

std::mutex account_lock;


void get_money(int id, double widthrawal)
{
	// EXCEPTION SAFE
	std::lock_guard<std::mutex> lock(account_lock);
	std::this_thread::sleep_for(std::chrono::seconds(5));
	std::cout << id << " tries to withdraw $" << widthrawal <<
		" on " << get_time() << "\n";
	if ((balance - widthrawal) >= 0)
	{
		balance -= widthrawal;
		std::cout << "New account balance is $" <<
			balance << "\n";
	}
	else
	{
		std::cout << "Not enough money in account.\n";
		std::cout << "Current balance is $" <<
			balance << "\n";
	}

}


std::mutex vec_lock;
std::vector<unsigned int> prime_vec;


void findprimes(unsigned int start, unsigned int end) //, std::vector<unsigned int>& vec)
{
	for (unsigned int x = start; x < end; x+=2)
	{
		for (unsigned int y = 2; y < x; y++)
		{
			if ((x% y) == 0)
			{
				break;
			}
			else if ((y+1) == x)
			{
				//vec.push_back(x);
				vec_lock.lock();
				prime_vec.push_back(x);
				vec_lock.unlock();
			}
		}
	}
}


void find_primes_with_threads(unsigned int start, unsigned int end, unsigned int num_threads)
{
	std::vector<std::thread> threadvec;
	unsigned int thread_spread = end / num_threads;
	unsigned int new_end = start + thread_spread - 1;
	for (unsigned int x = 0; x < num_threads; x++)
	{
		threadvec.emplace_back(findprimes, start, end);
		start += thread_spread;
		new_end += thread_spread;
	}
	for (auto& t : threadvec)
	{
		t.join();
	}
}


int main()
{
	/*
	std::thread th1(execute_thread, 1);
	th1.join();
	std::thread th2(execute_thread, 2);
	th2.join();
	*/


	/*
	std::thread threads[10];
	for (int i = 0; i < 10; i++)
	{
		threads[i] = std::thread(get_money, i, 15);
	}
	for (int i = 0; i < 10; i++)
	{
		threads[i].join();
	}
	*/

	/*
	std::vector<unsigned int> prime_vec;
	int start_time = clock();
	findprimes(1, 100000, prime_vec);
	for (auto i : prime_vec)
	{
		std::cout << i << "\n";
	}
	int endtime = clock();
	std::cout << "Execution time : " <<
		(endtime - start_time) / double(CLOCKS_PER_SEC)
		<< std::endl;
	*/

	int start_time = clock();
	find_primes_with_threads(1, 100000, 4);
	int end_time = clock();
	for (auto i : prime_vec)
	{
		std::cout << i << "\n";
	}
	std::cout << "Execution time : " <<
		(end_time - start_time) / double(CLOCKS_PER_SEC) <<
		std::endl;
	return 0;
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
