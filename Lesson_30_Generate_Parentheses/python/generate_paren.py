class Solution(object):
    
    def is_valid_paren(self, paren: str):
        stack = []
        while(paren):
            c = paren.pop(0)
            if c == '(':
                stack.push(c)
            elif c == ')':
                if stack[-1] == '(':
                    stack.pop()
                else:
                    return False
        if stack == []:
            return True
        return False
    
    def make_mask(self, length: int):
        mask = ['0']*2*n
        return mask.join("")
    
    def add_one(self, mask):
        # Given '101', add 1 to it to get
        # '110'
        bin_val = bin(mask)+1
        mask = str(bin_val)
        return mask
    
    def generate_parentheses(self, n: int):
        result = []
        # TODO - make binary string
        mask=self.make_mask(2*n)

        # build and test strings
        for i in range(2*n):
            parens = ""
            for  j in range(len(mask)):
                if mask[j] == 0:
                    parens.append("(")
                else:
                    parens.append(")")
            if(self.is_valid_paren(parens)):
                result.append(parens)
            mask=self.add_one(mask)
        return result
    
    
n = 5
# print(Solution().generate_parentheses(n))


class real_solution(object):
    
    def v2(self, n):
        res = []
        def back_track(S: str, left: int, right: int):
            if len(S) == 2*n:
                res.append(S)
                return
            if left < n:
                back_track(S+'(', left+1, right)
            if left > right:
                back_track(S+')', left, right+1)
            
        back_track('', 0, 0)
        return res
    
print(real_solution().v2(n))