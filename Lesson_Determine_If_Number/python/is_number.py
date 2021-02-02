from enum import Enum
class Solution(object):
    def __init__(self):
        pass
    
    def is_number(self, s):
        
        if len(s) < 1:
            return False
        
        seen_e = False;
        e_ind = -1
        
        seen_dec = False
        dec_ind = -1
        
        any_nums = False
        for i, char in enumerate(s):
            
            # minus sign
            if char == "-":
                # only valid if in first pos or if after e
                if i == 0 or (seen_e == True and e_ind == i-1):
                    continue
                else:
                    # print([i, e_ind, char])
                    return False
                
            # e sign
            # can only be after pos 0, can occur only once, need to at least have had numbers before
            elif char.lower() == "e":
                # print("seen_e, i, any_nums")
                # print([seen_e, i, any_nums])
                if (not seen_e) and (i != 0) and (any_nums == True):
                    seen_e = True
                    e_ind = i
                    continue
                else:
                    return False
                
            # decimal sign
            # can't follow after e, can only be one at most
            elif char == ".":
                if (not seen_dec) and (not seen_e):
                    seen_dec=True
                    dec_ind = i
                    continue
                else:
                    return False
            
            # is not numeric
            elif char.isdigit() == True:
                any_nums = True
                continue
            
            # exit loop if we get here
            else:
                return False
        
        if any_nums:
            return True
        return False
        
class DigitState(Enum):
    BEGIN = 0
    NEGATIVE1 = 1
    DIGIT1 = 2
    DOT = 3
    DIGIT2 = 4
    E = 5
    NEGATIVE2 = 6
    DIGIT3 = 7
    
NEXT_STATES_MAP = {
    DigitState.BEGIN: [DigitState.NEGATIVE1, DigitState.DIGIT1],
    DigitState.NEGATIVE1: [DigitState.DIGIT1, DigitState.DOT],
    DigitState.DIGIT1: [DigitState.DIGIT1, DigitState.DOT, DigitState.E],
    DigitState.DOT: [DigitState.DIGIT2],
    DigitState.E: [DigitState.NEGATIVE2, DigitState.DIGIT3],
    DigitState.DIGIT2: [DigitState.DIGIT2, DigitState.E],
    DigitState.NEGATIVE2: [DigitState.DIGIT3],
    DigitState.DIGIT3: [DigitState.DIGIT3],
}
    
STATE_VALIDATOR = {
    DigitState.BEGIN: lambda x: True,
    DigitState.DIGIT: lambda x: x.isdigit(),
    DigitState.NEGATIVE1: lambda x: x == '-',
    DigitState.DIGIT2: lambda x: x.isdigit(),
    DigitState.DOT: lambda x: x == '.',
    DigitState.E: lambda x: x == 'e',
    DigitState.NEGATIVE2: lambda x: x == '-',
    DigitState.DIGIT3: lambda x: x.isdigit()
}
    
    
def parse_number(in_str):
    state = DigitState.BEGIN
    for c in in_str:
        for next_state in NEXT_STATES_MAP[state]:
            if STATE_VALIDATOR[next_state](c):
                state = next_state
                found = True
                break
        if not found:
            return False
    
    # Got to end, need to make sure we are in good end state  
    return state in [DigitState.DIGIT1, DigitState.DIGIT2, DigitState.DIGIT3]
    
          
nums= ["123", "12.3","-123", "-.3", "1.5e5", "12a", "1.1234e-50", "1.1e--5"]

for num in nums:
    print(Solution().is_number(num))                