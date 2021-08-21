def first_recurring_character(in_str: str):
    seen = set()
    for c in in_str:
        if c in seen:
            return c
        seen.add(c)
    return None


print(first_recurring_character('qwertty'))
print(first_recurring_character('qwerty'))