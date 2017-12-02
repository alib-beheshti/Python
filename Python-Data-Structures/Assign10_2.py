fname = input("Enter file name: ")
fh = open(fname)
time_dic = dict()
for line in fh:
    if not line.startswith('From '):continue
    line_splt = line.split()
    time_val = line_splt[5]
    time_vec = time_val.split(':')
    time_dic[time_vec[0]] = time_dic.get(time_vec[0],0)+1
tup_list_sort = sorted(time_dic.items())
for k,v in tup_list_sort:
    print(k,v) 
