import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyntcloud import PyntCloud
import open3d as o3d
import plyfile
import time

# class UserMinMaxScaler:
#     def __init__(self):
#         self.max_num = -np.inf
#         self.min_num = np.inf

#     def fit(self, arr):
#         if arr is None:
#             print("fit() missing 1 required positional argument: 'X'")

#         self.max_num = np.min(arr)
#         self.min_num = np.max(arr)

#     def fit_transform(self, arr):
#         if arr is None:
#             print("fit_transform() missing 1 required positional argument: 'X'")

#         self.max_num = np.max(arr)
#         self.min_num = np.min(arr)

#         return (arr - self.min_num) / (self.max_num - self.min_num)

#     def transform(self, arr):
#         return (arr - self.min_num) / (self.max_num - self.min_num)
 

# plydata = plyfile.PlyData.read('scene0000_00_vh_clean_2.ply')
# UMM = UserMinMaxScaler()
# len = 10000
# data = torch.load('scene0000_00z.pth')
# coord = data["coord"][:len]
# color = data["color"][:len]
# normal = data["z_normal"][:len]

# tmp = pd.DataFrame(plydata['vertex'].data).values[:len]
# tmp[:,3:6] = normal
# list = []
# for i in tmp:
#     list.append(tuple(i))

# plydata['vertex'].data = np.array(list,dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')
#                               , ('red', 'f4'), ('green', 'f4'), ('blue', 'f4'), ('alpha', 'f4')])

# plydata.write('my_pts4.ply')


def div_z(data, sort_std, i):
    data = sorted(data, key = lambda x : x[int(sort_std[i,0])])
    div_list_a_split = data[:,int(sort_std[i,0])]
    return np.split(data, np.searchsorted(div_list_a_split, [1e-8]))

plydata = plyfile.PlyData.read('scene0000_00_vh_clean_2.ply')
data = torch.load('scene0000_00z.pth')
coord = data["coord"].tolist()
color = data["color"].tolist()
normal = data["z_normal"].tolist()

start = time.time()
x, y, z = np.std(coord[:,0]),np.std(coord[:,1]),np.std(coord[:,2])
std = [[0,x], [1,y], [2,z]]
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


print(f"{end - start:.5f} sec")
exit()


# def f_z_score(data): # z-score 함수 생성
#     mean = np.mean(data) #평균
#     std = np.std(data)   #표준편차
#     z_scores = [(y-mean)/std for y in data] #z-score
#     return z_scores

# z_scores = []
# for i in range(3):
#     z_score = f_z_score(data['coord'][:,i])
#     if z_scores == []:
#         for k in z_score:
#             z_scores.append([k])    
#     else:
#         for j,v in enumerate(z_score):
#             z_scores[j].append(v)
            
# print(np.array(z_scores).shape,z_scores[0])
# z_scores = np.array(z_scores)
# # df_list = pd.DataFrame(z_scores[:,2])
# # print(model['normal'][:,0])
# df_list = pd.DataFrame(model['normal'][:,2])

# plt.hist(df_list,bins = 20)

# plt.show()