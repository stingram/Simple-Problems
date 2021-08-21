#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include <numeric>
#include <sstream>
#include <algorithm>


int main() {
 std::cout << "abs(-10) = " << std::abs(-10) << "\n";
    
    std::cout << "max(5,4) = " << std::max(5,4) << "\n";
    
    std::cout << "min(5,4) = " << std::min(5,4) << "\n";
    
    std::cout << "fmax(5.3,4.3) = " << std::fmax(5.3,4.3) << "\n";
    
    std::cout << "fmin(5.3,4.3) = " << std::fmin(5.3,4.3) << "\n";
    
    // e ^ x
    std::cout << "exp(1) = " << std::exp(1) << "\n";
    
    // 2 ^ x
    std::cout << "exp2(1) = " << std::exp2(1) << "\n";
    
    // e * e * e ~= 20 so log(20.079) ~= 3
    std::cout << "log(20.079) = " << std::log(20.079) << "\n";
    
    // 10 * 10 * 10 = 1000, so log10(1000) = 3
    std::cout << "log10(1000) = " << std::log10(1000) 
            << "\n";
    
    // 2 * 2 * 2 = 8
    std::cout << "log2(8) = " << std::log2(8) 
            << "\n";
    
    // 2 ^ 3
    std::cout << "pow(2,3) = " << std::pow(2,3) 
            << "\n";
    
    // Returns what times itself equals the provided value
    std::cout << "sqrt(100) = " << std::sqrt(100) 
            << "\n";
    
    // What cubed equals the provided
    std::cout << "cbrt(1000) = " << std::cbrt(1000) 
            << "\n";
    
    // Hypotenuse : SQRT(A^2 + B^2)
    std::cout << "hypot(2,3) = " << std::hypot(2,3) 
            << "\n";
    
    std::cout << "ceil(10.45) = " << std::ceil(10.45) 
            << "\n";
    
    std::cout << "floor(10.45) = " << std::floor(10.45) 
            << "\n";
    
    std::cout << "round(10.45) = " << std::round(10.45) 
            << "\n";
    
    // Also sin, cos, tan, asin, acos, atan, atan2,
    // sinh, cosh, tanh, asinh, acosh, atanh
    
    return 0;
}