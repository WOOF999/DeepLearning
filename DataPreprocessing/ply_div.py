import torch
import numpy as np
import time

def div_z(data, sort_std, i):
    data = sorted(data, key = lambda x : x[int(sort_std[i,0])])
    div_list_a_split=np.array(data).T[int(sort_std[i,0])]
    return np.split(data, np.searchsorted(div_list_a_split, [1e-8]))

data = torch.load('scene0000_00z.pth')
coord = data["coord"].tolist()
normal = data["z_normal"].tolist()

start = time.time()
std_col=np.std(coord,axis=0)

std = [[0,std_col[0]], [1,std_col[1]], [2,std_col[2]]]
sort_std = np.array(sorted(std,key = lambda x : x[1], reverse=True))
list_a_split = normal


for i in range(3):
    tmp = []
    if i ==0:
        list_a_split = div_z(list_a_split,sort_std, i)
    else:
        for data in list_a_split:
            temp = div_z(data,sort_std, i)
            for k in temp:
                tmp.append(k)
        list_a_split = tmp

end = time.time()
print(list_a_split)

print(f"{end - start:.5f} sec")
exit()


