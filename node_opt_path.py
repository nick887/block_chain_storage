# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 15:00:51 2023
node optimal path
@author: Lethe
"""
import numpy as np


#floyd求最短路径函数
def floyd_shortest_paths(distance, path):
    nodeNum = len(distance)
    for k in range(nodeNum):
        for i in range(nodeNum):
            for j in range(i,nodeNum):
                if distance[i][j] > distance[i][k]+distance[k][j]:
                    distance[i][j] = distance[i][k]+distance[k][j]
                    distance[j][i] = distance[i][k]+distance[k][j]
                    path[i][j]=k
                    path[j][i]=k
    return distance, path

#解析最优路径 返回一个list
def trans_path(startP, endP, path):
    route = [endP]
    point = path[startP][endP]
    
    if point == -2:
        print('路径未连通')
        return route
    elif point == -1:  
        return route
    else:
        tempRoute =  trans_path(startP,point,path)
        route = route + tempRoute
        return route
     

nodeNum = 30
effectScale = 30
xscale = 100 #节点空间范围-x
yscale = 100 #节点空间范围-y

data = np.load('./Data/NodeInfo/nodeGraphInfo-{}-{}-{}-{}.npz'.format(nodeNum,effectScale,xscale,yscale))
nodePos,nodeDirect,nodeDistance = data['arr_0'],data['arr_1'],data['arr_2']

#nodeOptDis = nodeDistance #最优距离
nodePath = nodeDirect-2 #-1代表直连 -2代表未联通
nodePath = nodePath.astype(int)

nodeOptDis,nodePath = floyd_shortest_paths(nodeDistance,nodePath)

'''nodeOptDis 节点间最近距离 nodePath 节点最短路径'''
np.savez('./Data/NodeInfo/nodeRouteInfo-{}-{}-{}-{}.npz'.format(nodeNum,effectScale,xscale,yscale)
         ,nodeOptDis,nodePath)

#%%
#输出所有的路径
for startP in range(nodeNum):
    print('******************')
    for endP in range(startP+1,nodeNum):
        route = trans_path(startP,endP,nodePath) + [startP]
        route.reverse()
        print('node {} to node {} path is: {}'.format(startP,endP,route))
