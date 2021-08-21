#include <string>
#include <iostream>

template <typename T>
void swap(T &a, T &b)
{
    T temp = a;
    a = b;
    b = temp;

    // std::cout << "a: " << a << "\tb: " << b << "\n";
}

template<typename T>
void swap(T a[], T b[], int size)
{
    for(int i=0;i<size;i++)
    {
        T temp = a[i];
        a[i] = b[i];
        b[i] = temp;
    }
}


int main()
{
    int a = 10;
    int b = 20;

    swap(a,b);
    std::cout << "a: " << a << "\tb: " << b << "\n";

    std::string fn = "Caleb";
    std::string ln = "Curry";

    swap(fn, ln);
    std::cout << "fn: " << fn << "\tln: " << ln << "\n";

    int const SIZE = 6;
    int nines[] = {9,9,9,9,9,9};
    int ones[] = {1,1,1,1,1,1};

    for(int i=0;i<SIZE;i++)
    {
        std::cout << nines[i] << " " << ones[i] << "\t";
    }
    std::cout << "\n\n";

    swap(nines, ones, 6);

    for(int i=0;i<SIZE;i++)
    {
        std::cout << nines[i] << " " << ones[i] << "\t";
    }
    std::cout << "\n\n";
    
    return 0;
}