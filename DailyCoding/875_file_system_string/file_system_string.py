# "dir\n\tsubdir1\n\tsubdir2\n\t\tfile.ext"
# assume always start with a dir


def found_newline(s,index):
    while index < len(s):
        if s[index] == '\n':
            return True, index+1
        index += 1
    return False, -1

def count_tabs(s,index):
    num_tabs = 0
    found_a_tab = False
    while index <len(s):
        if s[index] == '\t':
            num_tabs+=1
            post_tabs_index = index+1
            found_a_tab = True
        elif s[index] != '\t' and found_a_tab:
            return num_tabs, post_tabs_index
        index += 1
    return num_tabs, post_tabs_index

def end_of_filename(s,index):
    while index < len(s):
        if s[index] == '\\':
                return index
        index += 1
    return index

def next_period_or_new_line(s,index):
    is_new_directory = True
    while index < len(s):
        if s[index] == '.':
            # found file
            new_index = end_of_filename(s,index)
            return new_index,False
        if index < len(s):
            if s[index] == '\n':
                return index, is_new_directory 
        index += 1
    return -1, False

def longest_path_length(file_system: str):
    longest = 3
    index = 3 # assume we at least have dir
    prefixes = {'dir': 3}
    curr_str = "dir"
    while index < len(file_system):
        # consume next directory or file
        newline_found, index = found_newline(file_system,index)
        print(f"newline_found:{newline_found}, index:{index}")
        if newline_found:
            tabs, index = count_tabs(file_system,index)
            print(f"tabs:{tabs},index:{index}")
            new_index, new_directory = next_period_or_new_line(file_system,index)
            print(f"new_index:{new_index},new_directory:{new_directory}")
            if new_directory:
                prefixes[file_system[:new_index]] =(new_index+1,new_index-index)
            else:
                # print(f"")
                prefixes[file_system[:new_index]] =(index+1,new_index-index)
                if new_index+1 > longest:
                    longest = new_index+1
            index = new_index
        else:
            break
    return longest

s = "dir\n\tsubdir1\n\tsubdir2\n\t\tfile.ext"
print(f"s: {s}")
print(f"{longest_path_length(s)}")