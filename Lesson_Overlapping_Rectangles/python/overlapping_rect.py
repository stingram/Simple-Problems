class Rectangle(object):
    def __init__(self, min_x, min_y, max_x, max_y):
        self.min_x = min_x
        self.min_y = min_y
        self.max_x = max_x
        self.max_y = max_y
        
    def area(self):
        return (self.max_x-self.min_x)*(self.max_y-self.min_y)
    

def overlap_area(a: Rectangle, b: Rectangle):
    min_x = max(a.min_x, b.min_x)
    min_y = max(a.min_y, b.min_y)
    
    max_x = min(a.max_x, b.max_x)
    max_y = min(a.max_y, b.max_y)
    
    return Rectangle(min_x, min_y, max_x, max_y).area()


a = Rectangle(1,1,3,2)
b = Rectangle(2,1,4,5)

print(overlap_area(a,b))