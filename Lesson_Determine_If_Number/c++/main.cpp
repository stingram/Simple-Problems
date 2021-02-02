#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <map>
#include <unordered_map>
#include <string>
#include <sstream>
#include <functional>

// class DigitState(Enum):
    // BEGIN = 0
    // NEGATIVE1 = 1
    // DIGIT1 = 2
    // DOT = 3
    // DIGIT2 = 4
    // E = 5
    // NEGATIVE2 = 6
    // DIGIT3 = 7
    
// NEXT_STATES_MAP = {
    // DigitState.BEGIN: [DigitState.NEGATIVE1, DigitState.DIGIT1],
    // DigitState.NEGATIVE1: [DigitState.DIGIT1, DigitState.DOT],
    // DigitState.DIGIT1: [DigitState.DIGIT1, DigitState.DOT, DigitState.E],
    // DigitState.DOT: [DigitState.DIGIT2],
    // DigitState.E: [DigitState.NEGATIVE2, DigitState.DIGIT3],
    // DigitState.DIGIT2: [DigitState.DIGIT2, DigitState.E],
    // DigitState.NEGATIVE2: [DigitState.DIGIT3],
    // DigitState.DIGIT3: [DigitState.DIGIT3],
// }
    
// STATE_VALIDATOR = {
//     DigitState.BEGIN: lambda x: True,
//     DigitState.DIGIT: lambda x: x.isdigit(),
//     DigitState.NEGATIVE1: lambda x: x == '-',
//     DigitState.DIGIT2: lambda x: x.isdigit(),
//     DigitState.DOT: lambda x: x == '.',
//     DigitState.E: lambda x: x == 'e',
//     DigitState.NEGATIVE2: lambda x: x == '-',
//     DigitState.DIGIT3: lambda x: x.isdigit()
// }
    
    
// def parse_number(in_str):
//     state = DigitState.BEGIN
//     for c in in_str:
//         for next_state in NEXT_STATES_MAP[state]:
//             if STATE_VALIDATOR[next_state](c):
//                 state = next_state
//                 found = True
//                 break
//         if not found:
//             return False
    
//     # Got to end, need to make sure we are in good end state  
//     return state in [DigitState.DIGIT1, DigitState.DIGIT2, DigitState.DIGIT3]
    

enum class DigitState
{
    BEGIN = 0,
    NEGATIVE1 = 1,
    DIGIT1 = 2,
    DOT = 3,
    DIGIT2 = 4,
    E = 5,
    NEGATIVE2 = 6,
    DIGIT3 = 7, 
};

std::unordered_map<DigitState,std::vector<DigitState>> next_state_map= 
{
    {DigitState::BEGIN, std::vector<DigitState>{DigitState::NEGATIVE1, DigitState::DIGIT1}},
    {DigitState::NEGATIVE1, std::vector<DigitState>{DigitState::DIGIT1, DigitState::DOT}},
    {DigitState::DIGIT1, std::vector<DigitState>{DigitState::DIGIT1, DigitState::DOT, DigitState::E}},
    {DigitState::DOT, std::vector<DigitState>{DigitState::DIGIT2}},
    {DigitState::E, std::vector<DigitState>{DigitState::NEGATIVE2, DigitState::DIGIT3}},
    {DigitState::DIGIT2, std::vector<DigitState>{DigitState::DIGIT2, DigitState::E}},
    {DigitState::NEGATIVE2, std::vector<DigitState>{DigitState::DIGIT3}},
    {DigitState::DIGIT3, std::vector<DigitState>{DigitState::DIGIT3}},
};

std::unordered_map<DigitState, std::function<bool(const char&)>> STATE_VALIDATOR = {
    {DigitState::BEGIN, [](const char& x) {return true;}},
    {DigitState::DIGIT1, [](const char& x) {return ::isdigit(x);}},
    {DigitState::NEGATIVE1, [](const char& x) {return x == '-';}},
    {DigitState::DIGIT2, [](const char& x) {return ::isdigit(x);}},
    {DigitState::DOT, [](const char& x) {return x == '.';}},
    {DigitState::E, [](const char& x) {return x == 'e';}},
    {DigitState::NEGATIVE2, [](const char& x) {return x == '-';}},
    {DigitState::DIGIT3, [](const char& x) {return ::isdigit(x);}},
};

bool parse_number(std::string in_str)
{
    DigitState state = DigitState::BEGIN;
    DigitState next_state;
    bool found = false;
    std::vector<DigitState> valid_end_states = {DigitState::DIGIT1,
                                                DigitState::DIGIT2,
                                                DigitState::DIGIT3};

    for(const char & c : in_str)
    {
        for(DigitState next_state: next_state_map[state])
        {
            if(STATE_VALIDATOR[next_state](c))
            {
                state = next_state;
                found = true;
                break;
            }
        }
    }
    if(!found)
    {
        return false;
    }
    return std::find(valid_end_states.begin(), valid_end_states.end(), state) != valid_end_states.end();


}

int main()
{
    std::cout << parse_number("123e5") << "\n";
    return 0;
}