# "dir\n\tsubdir1\n\tsubdir2\n\t\tfile.ext"
# assume always start with a dir

def longest_path_length(file_system: str):
    longest = 0
    paths = {-1:0}
    for line in file_system.splitlines():
        name = line.lstrip('\t')
        depth = len(line) -len(name)
        if '.' in name:
            longest = max(longest,paths[depth-1]+len(name))
        else:
            paths[depth] = paths[depth-1]+len(name)+1
    return longest

s = "dir\n\tsubdir1\n\tsubdir2\n\t\tfile.ext"
print(f"s: {s}")
print(f"{longest_path_length(s)}")


s = "dir\n\tsubdir1\n\t\tfile1.ext\n\t\tsubsubdir1\n\tsubdir2\n\t\tsubsubdir2\n\t\t\tfile2.ext" # 32
print(f"s: {s}")
print(f"{longest_path_length(s)}")