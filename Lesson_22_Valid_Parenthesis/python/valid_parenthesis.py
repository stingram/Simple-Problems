class Solution:
    def is_valid(self, s):
        stack = []
        
        # loop through string
        for c in s:
            # check for ()
            if c == '(':
                stack.append(c)
            if c == ')':
                # can't be valid because the first character was )
                if len(stack) == 0:
                    return False
                # can't be valid if we can't match with opening (
                if stack[-1] != '(':
                    return False
                # since we got a matching ), we can pop the (
                else:
                    stack.pop()
            
            # check for {}
            if c == '{':
                stack.append(c)
            if c == '}':
                # can't be valid because the first character was }
                if len(stack) == 0:
                    return False
                # can't be valid if we can't match with opening {
                if stack[-1] != '{':
                    return False
                # since we got a matching }, we can pop the {
                else:
                    stack.pop()
            
            # check for []
            if c == '[':
                stack.append(c)
            if c == ']':
                # can't be valid because the first character was ]
                if len(stack) == 0:
                    return False
                # can't be valid if we can't match with opening [
                if stack[-1] != '[':
                    return False
                # since we got a matching ], we can pop the [
                else:
                    stack.pop()
            
            print("STACK: {}".format(stack))
            
        # check if stack is empty
        if len(stack) > 0:
            return False
        return True
    
    
input = "[[{}()]]"
print(Solution().is_valid(input))