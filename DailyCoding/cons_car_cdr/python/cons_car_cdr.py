# cons(a, b) constructs a pair, and car(pair) and cdr(pair) returns the first and last element of that pair.
# For example, car(cons(3, 4)) returns 3, and cdr(cons(3, 4)) returns 4.

# GIVEN
def cons(c, d):
    def pair(f):
        return f(c, d)
    return pair


# Implement car and cdr.
def car(f):
    def return_first(a,b):
        return a
    return f(return_first)
print(f"car: {car(cons(3,4))}.")

def cdr(f):
    def return_last(a,b):               # this works only because cons defined about has two arguments that are saved and ready to be processed
        return b                        # Note that even though c and d variables are used in cons, I can use a and b here
    return f(return_last)
print(f"cdr: {cdr(cons(3,4))}.")

# cons returns a function that will accept a function that operates on the two arguments passed to cons
# so then if I want a function (called car) that takes cons function as an input, I need to return the result
# of passing a new function to the cons function that was given to my car function


