fname = input("Enter file name: ")
fh = open(fname)
address = dict()
for line in fh:
    if not line.startswith('From '):continue
    line_splt = line.split()
    address[line_splt[1]] = address.get(line_splt[1],0)+1
big_count = None
big_name = None
for name,count in address.items():
    if big_count == None or count > big_count:
        big_count = count
        big_name = name
print(big_name,big_count)
