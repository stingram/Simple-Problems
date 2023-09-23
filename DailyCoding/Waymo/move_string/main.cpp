#include <string>
#include <iostream>
#include <vector>
#include <cstring>


class MyString {
public:
    // default constructor
    MyString(const char* str)
    {
        size_ = std::strlen(str);
        data_ = new char[size_+1];
        std::strcpy(data_,str);
    }
    // Move constructor
    MyString(MyString&& other) noexcept {
        data_ = other.data_;
        size_ = other.size_;
        // Reset the source object
        other.data_ = nullptr;
        other.size_ = 0;
    }

    // deconstructor
    ~MyString() {
        delete[] data_;
    }
    
private:
    char* data_;
    size_t size_;
};

int main()
{

    return 0;
}