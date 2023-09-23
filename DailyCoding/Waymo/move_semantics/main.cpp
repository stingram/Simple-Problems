#include <iostream>
#include <vector>
#include <chrono>


void practice_chrono(void)
{
    auto start = std::chrono::high_resolution_clock::now();
    int x;
    x = 5;
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Duration: " << duration.count() << ".\n";
}

// Function to copy a large vector
std::vector<int> CopyVector(const std::vector<int>& source) {
    return source; // Copy semantics
}

// Function to move a large vector
std::vector<int> MoveVector(std::vector<int>&& source) {
    return std::move(source); // Move semantics
}

int main() {
    const int dataSize = 10000000; // Size of the data (1 million integers)

    // Create a large vector with data
    std::vector<int> sourceVector(dataSize, 42);

    // Measure the time taken to copy the vector
    auto startCopy = std::chrono::high_resolution_clock::now();
    std::vector<int> copiedVector = CopyVector(sourceVector);

    auto endCopy = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> copyDuration = endCopy - startCopy;

    // Measure the time taken to move the vector
    auto startMove = std::chrono::high_resolution_clock::now();
    std::vector<int> movedVector = MoveVector(std::move(sourceVector));

    auto endMove = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> moveDuration = endMove - startMove;

    std::cout << "Time taken for copying: " << copyDuration.count() << " seconds." << std::endl;
    std::cout << "Time taken for moving: " << moveDuration.count() << " seconds." << std::endl;

    return 0;
}