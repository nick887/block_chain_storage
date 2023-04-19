# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 14:45:53 2023
节点图
@author: Lethe
"""
import numpy as np
import matplotlib.pyplot as plt

def distance(node1,node2):
    return ((node1[0]-node2[0])**2+(node1[1]-node2[1])**2)**0.5



np.random.seed(2023)

nodeNum = 30
effectScale = 30
xscale = 100 #节点空间范围-x
yscale = 100 #节点空间范围-y

node_x = np.random.randint(0,xscale,(nodeNum))
node_y = np.random.randint(0,yscale,(nodeNum))

# node_x = np.random.randn(nodeNum)*20 +80
# node_y = np.random.randn(nodeNum)*20 +80

#节点坐标
#nodePos = np.vstack((np.arange(nodeNum),node_x,node_y)).T
nodePos = np.vstack((node_x+node_y,node_x,node_y)).T

nodePos = nodePos[np.argsort(nodePos[:,0])]
nodePos[:,0] =np.arange(nodeNum)
#节点直连矩阵
nodeDirect = np.zeros((nodeNum,nodeNum))
#节点距离矩阵
nodeDistance = np.zeros((nodeNum,nodeNum),dtype=int)
'''无穷大用10000表示'''
nodeDistance.fill(10000) 

for index in range(nodeNum):
    x = nodePos[index,1]
    y = nodePos[index,2]
    nodeNeibor = (nodePos[:,1]<=x+effectScale)&(nodePos[:,1]>=x-effectScale)&(nodePos[:,2]<=y+effectScale)&(nodePos[:,2]>=y-effectScale)
    nodeNeibor = nodePos[nodeNeibor,0].astype(int)
    for node in nodeNeibor:
        nodeDis = distance(nodePos[index,1:],nodePos[node,1:])
        if(nodeDis<=effectScale):
            nodeDirect[index,node] = 1
            nodeDirect[node,index] = 1
            nodeDistance[index,node] = int(nodeDis)
            nodeDistance[node,index] = int(nodeDis)

'''nodePos 节点坐标 nodeDirect 节点直接连接状况 nodeDistance 节点距离'''
np.savez('./Data/NodeInfo/nodeGraphInfo-{}-{}-{}-{}'.format(nodeNum,effectScale,xscale,yscale),
         nodePos,nodeDirect,nodeDistance)

#%%
#节点init NEW
nodeLimit = np.random.rand(nodeNum)*500+200 #节点的存储空间
np.savetxt('./Data/NodeInfo/nodeLimit-'+str(nodeNum)+'.csv', nodeLimit, delimiter=',')


#%%
#画图
fig = plt.figure(figsize=(6,6), dpi= 300)

for index in range(nodeNum):
    for node in range(index+1,nodeNum):
        if(bool(nodeDirect[node,index])):
            plt.plot((nodePos[node,1],nodePos[index,1]),(nodePos[node,2],nodePos[index,2]),'r')

plt.scatter(nodePos[:,1],nodePos[:,2],s=100,marker='v')
fig.savefig('./Data/NodeInfo/nodeGraph-'+str(nodeNum)+'.png')




