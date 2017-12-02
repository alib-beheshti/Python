fname = input("Enter file name: ")
fh = open(fname)
count = 0
for line in fh:
    if not line.startswith('From '):continue
    line_splt = line.split()
    count = count+1
    print(line_splt[1])
print("There were", count, "lines in the file with From as the first word")
