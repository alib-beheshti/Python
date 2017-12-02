fname = input("Enter file name: ")
fh = open(fname)
lst = list()
for line in fh:
    line_strp = line.rstrip()
    words = line_strp.split()
    for word in words:
        if word in lst: continue
        lst.append(word)
lst.sort()
print(lst)
