# Use words.txt as the file name
fname = input("Enter file name: ")
fhand = open(fname)
data = fhand.read()
data_upper = data.upper()
data_print = data_upper.rstrip()
print(data_print)
