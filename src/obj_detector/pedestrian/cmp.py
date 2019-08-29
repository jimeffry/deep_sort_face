import numpy as np
def compare_2(str_a,str_b):
    num = min(len(str_a),len(str_b))
    idx = 0
    str_out = str_a if len(str_a)>len(str_b) else str_b
    while idx < num:
        if str_a[idx] > str_b[idx]:
            str_out = str_a
            break
        elif str_a[idx]< str_b[idx]:
            str_out = str_b
            break
        else:
            idx+=1
    return str_out

def main(str_cmp):
    str_cmp_split = str_cmp.strip().split(',')
    for i in range(len(str_cmp_split)-1):
        for j in range(i+1,len(str_cmp_split)):
            str_tmp = compare_2(str_cmp_split[i],str_cmp_split[j])
            if str_tmp != str_cmp_split[i]:
                str_cmp_split[j] = str_cmp_split[i]
                str_cmp_split[i] = str_tmp
    return str_cmp_split

if __name__=='__main__':
    test_s = "stry,zre,abr,abrc,aqc,wxy,erc,fgt, ,"
    rt = main(test_s)
    rt = ','.join(rt)
    print(rt)
