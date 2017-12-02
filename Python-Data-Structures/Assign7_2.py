# Use the file name mbox-short.txt as the file name
fname = input("Enter file name: ")
fh = open(fname)
count = 0
tot = 0
for line in fh:
    if not line.startswith("X-DSPAM-Confidence:") : continue
    count = count+1
    space_ind = line.find(':')
    new_ind = line.find('\n')
    num_chr = line[space_ind+1:new_ind]
    #print(num_chr)
    tot=tot+float(num_chr)
print("Average spam confidence:",tot/count)
