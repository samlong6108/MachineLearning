from scipy.special import comb
import math
#---------------------------------------
prev_a,prev_b=10,1
p=0
#----------------------------------------
def calculate_ab(_list,a,b):
    cur_a,cur_b=0,0
    length=len(_list)
    _list=list(_list)
    for i in range(length):
        if _list[i]== '1':
            a+=1
            cur_a+=1
        else:
            b+=1
            cur_b+=1
    return (a,b,cur_a,cur_b)

#-----------------------------------------
fp=open('./testfile.txt')
line = fp.readline()
content = []
while(line):
    line=line.strip('\n')
    content.append(line)
    line=fp.readline()
fp.close
txt_length=len(content)
#-----------------------------------------
for i in range (txt_length):
    print('case%d: %s' %(i+1,content[i]))
    post_a,post_b,cur_a,cur_b=calculate_ab(content[i],prev_a,prev_b)
    p=cur_a/(cur_a+cur_b)
    likelihood = comb(cur_a+cur_b,cur_a)*(p**cur_a)*((1-p)**cur_b)
    print('Likelihood: %.17f' %(likelihood))
    print('Beta Prior: a=%d b=%d' %(prev_a,prev_b))
    prev_a,prev_b=post_a,post_b
    print('Beta Postirior: a=%d b=%d' %(post_a,post_b))
    print("")