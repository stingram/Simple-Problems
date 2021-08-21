import os

def clean_path(raw_path):
    cleaned = raw_path.split("/")
    stack = []
    for p in cleaned:
        if p == '.':
            pass
        elif p == '..':
            stack.pop()
        else:
            stack.append(p)
    new_path = '/'.join(stack)
    return new_path


path = '/users/tech/docs/.././desk/../'
print(clean_path(path))
# /users/tech/