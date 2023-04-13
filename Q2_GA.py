# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 10:51:59 2022
Q2-GA-opt2
@author: Lethe
"""

import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib import rcParams

np.random.seed(1037)
random.seed(1037)

'''随机生成区块 按照备份数量生成'''
def generate_blockAllo_backups(blockBackupsUpper,alloNum=10):
    blockAlloGen = np.zeros((alloNum,nodeNum,blockNum)).astype(bool)
    blockBackupsUpper = blockBackupsUpper+1
    #blockBackupsUpper = int(blockBackups*2)
    for i in range(alloNum):
        blockAllo = np.zeros((nodeNum,blockNum)).astype(bool)
        while(np.all(constraint(blockAllo))==False):
            blockAllo = np.zeros((nodeNum,blockNum)).astype(bool)
            for block in range(blockNum):
                nodeChoice = random.sample(range(nodeNum), np.random.randint(blockBackups,blockBackupsUpper))
                blockAllo[nodeChoice,block] = True
            #print(constraint(blockAllo))
        blockAlloGen[i,:,:] = blockAllo
        
    return blockAlloGen

'''
约束条件判断
输入 区块分配方案 blockAllo
输出 True False
其中 blockInfo blockBackups nodeLimit storageLimit 为全局变量,条件可选
'''
def constraint(blockAllo, condition = -1): #-1代表所有条件,0-2代表条件1-3
    if(condition == -1):
        constraint_result = np.full(3, False, dtype=bool)  #三个约束的结果 默认全False
        #约束条件1 每个节点的空间限制
        constraint_result[0] = np.all(np.dot(blockAllo,blockInfo[:,1]) <= nodeLimit) #为True即为节点均满足要求
        #约束条件2 每个区块至少有blockBackups备份
        constraint_result[1] = np.all(blockAllo.sum(axis=0) >= blockBackups) #为True即为满足备份数量要求
        #约束条件3 系统总的存储空间占用率不超过限定值
        constraint_result[2] = (np.sum(blockAllo.sum(axis=0)*blockInfo[:,1])<=storageLimitcondition) #为True即为满足约束
        return constraint_result
    
    elif(condition == 0):
        return np.all(np.dot(blockAllo,blockInfo[:,1]) <= nodeLimit)
    
    elif(condition == 1):
        return np.all(blockAllo.sum(axis=0) >= blockBackups)
    
    elif(condition == 2):
        return np.sum(blockAllo.sum(axis=0)*blockInfo[:,1])/np.sum(nodeLimit)<=storageLimit
    

'''
计算目标函数
全局变量 costAll
'''
def objective(blockAllo):
    nodeBlockCost = np.zeros((nodeNum,blockNum)) #节点访问区块的代价,行代表节点,列代表区块
    for node in range(nodeNum):
        for block in range(blockNum):
            nodeBlockCost[node,block] = np.min(costAll[block,blockAllo[:,block]==1,node]) #选出通信成本最小的节点所需成本
    nodeRatio = np.dot(blockAllo,blockInfo[:,1])/nodeLimit #节点存储空间占总空间的比例
    #nodeVar = np.var(nodeRatio)
    nodeProportion = (np.exp(5*nodeRatio)-1)/(np.exp(5)-1) #按照指定函数对空间占用率进行处理
    
    #分别为 通信成本 存储平衡度 总目标
    #计算总目标时乘以 1/区块数量
    nodeBlockCostAvg = np.sum(nodeBlockCost)/(nodeNum*blockNum)
    nodeProportionAvg = np.sum(nodeProportion)/nodeNum*50
    result = np.array([nodeBlockCostAvg+nodeProportionAvg*node_cost_ratio, nodeBlockCostAvg, nodeProportionAvg*node_cost_ratio])
    return result

'''计算个体的适应度'''
def cal_fitness(blockAlloGen):
    alloNum = blockAlloGen.shape[0]
    #行代表一种分配/个体 列依次为 适应度 总目标 通信成本 存储平衡度 
    alloTarget = np.zeros((alloNum,3)) 
    for index,blockAllo in enumerate(blockAlloGen): #遍历
        alloTarget[index,:] = objective(blockAllo)
    
    alloFit = -(alloTarget[:,0]-np.max(alloTarget[:,0]))+1e-5 #计算适应度 适应度为正数
    #alloFit[:,0] = np.exp(alloFit[:,0]*10)
    return alloFit,alloTarget

'''选择一定数量的个体'''
def select(alloFit, selectSize=1):
    #alloFitIndex = np.hstack(np.arange(len(alloFit)),alloFit)
    # sortIndex = np.argsort(-alloFit[:,0],axis=0) #按照适应度排序 降序
    # idx = np.zeros(selectSize,dtype=int)
    # topN = int(alloNum/5) #直接保留前N个数据
    # idx[0:topN] = sortIndex[0:topN]
    
    # idx[topN:] = np.random.choice(sortIndex[topN:], size=selectSize-topN, replace=False, #replace代表是否能重复抽取
    #                        p=(alloFit[sortIndex[topN:],0])/(alloFit[sortIndex[topN:],0].sum()) )
    
    idx = np.random.choice(np.arange(alloFit.shape[0]), size=selectSize, replace=False, #replace代表是否能重复抽取
                           p=(alloFit)/(alloFit.sum()) )
    
    return idx

'''交叉变异'''
def crossover_mutation(blockAlloGen, CROSSOVER_RATE=0.8, MUTATION_RATE=0.5):
    blockAlloNew = []
    alloSize = blockAlloGen.shape[0]
    crossover_flag, mutation_flag = 0,0
    for alloFa in blockAlloGen:		#遍历种群中的每一个个体，将该个体作为父亲
        blockAlloNew.append(alloFa)
        alloChild = alloFa.copy()		#孩子先得到父亲的全部基因
        #交叉
        if np.random.rand() < CROSSOVER_RATE:			#以一定的概率发生交叉
            crossover_flag = 1
            alloMa = blockAlloGen[np.random.randint(alloSize),:,:]	#再种群中选择另一个个体，并将该个体作为母亲
            alloChild2 = alloMa.copy()
            #crossPoints = np.random.randint(low=0, high=blockNum, size=nodeNum)	#随机产生交叉的点
            crossPoints = np.random.choice(np.arange(blockNum), size = int(nodeNum/2), replace=False)
            for i in crossPoints: #交换列
                alloChild[:,i] = alloMa[:,i] #孩子得到位于交叉点母亲的基因 列 
                alloChild2[:,i] = alloFa[:,i]
        #变异
        if np.random.rand() < MUTATION_RATE: 				#以MUTATION_RATE的概率进行变异
            mutation_flag = 1
            mutate_block = np.random.randint(0, blockNum)	#随机产生一个实数 代表要变异基因的位置/列
            mutate_node = np.random.randint(0, nodeNum)	#随机产生一个实数 代表要变异基因的位置/行
            alloChild[mutate_node, mutate_block] = alloChild[mutate_node, mutate_block]^1 	#将变异点的二进制为反转

        
        # if np.all(constraint(alloChild))==True:
        #     print('产生子代')
        #     blockAlloNew.append(alloChild) #若交叉后的个体不满足约束条件 则不进行交叉
        if(crossover_flag|mutation_flag):
            constraint_result = constraint(alloChild)
            #大多数是因为第1个条件不满足要求
            #print(constraint_result)
            if(constraint_result[1]==False):
                #print('修复')
                alloChild = fix_constraint1(alloChild)
                constraint_result = constraint(alloChild)
                #print(constraint_result)
            if np.all(constraint_result):

                blockAlloNew.append(alloChild)
                
            #孩子2    
            if(crossover_flag):    
                constraint_result = constraint(alloChild2)
                if(constraint_result[1]==False):
                    #print('修复')
                    alloChild = fix_constraint1(alloChild2)
                    constraint_result = constraint(alloChild2)
                    #print(constraint_result)
                if np.all(constraint_result):
    
                    blockAlloNew.append(alloChild2)
                
            crossover_flag, mutation_flag = 0,0

    return np.array(blockAlloNew)


'''
对交叉变异后不满足条件1的情况进行修复(0-2)
'''
def fix_constraint1(blockAllo):
    
    num = blockAllo.sum(axis=0) #该分配每个区块的数量
    for index,block in enumerate(num):
        if block<blockBackups:
            idx = np.random.choice(np.where(blockAllo[:,index]==False)[0], size=blockBackups-block, replace=False) #replace代表是否能重复抽取
            for i in idx:
                blockAllo[i,index] = True                            
    
    return blockAllo

if __name__=="__main__":
    '''初始输入信息'''
    blockNum = 100 #区块数量
    nodeNum = 30 #节点数量
    blockBackups = 4 #区块备份数量 至少为1
    storageLimit = 0.65 #优化后系统总存储空间占比
    storageLimitLoad = 0.5 #从Q1加载的限制区块数据量
    alloNum = 10 #种群数量
    epochNum = 1000 #迭代次数
    saveResult = True #是否保存数据 True False
    expfromQ1 = False
    node_cost_ratio = 1
    
    #区块信息
    blockInfo = np.loadtxt('./Data/blockInfo{}.csv'.format(blockNum), delimiter=',')
    
    #节点信息
    nodeLimit = np.loadtxt('./Data/NodeInfo/nodeLimit-{}.csv'.format(nodeNum), delimiter=',')
    nodeOptDis = np.load('./Data/NodeInfo/nodeRouteInfo-{}-30-100-100.npz'.format(nodeNum))['arr_0']
    nodeStorage = np.sum(nodeLimit) #CU本地存储的空间和
    storageLimitcondition = np.sum(nodeLimit)*storageLimit #CU所有节点的空间限制
    
    #storageLimitProPara = 
    
    print('Start...') 
    
    if(expfromQ1): #数据使用Q1选出的区块
        print('The experiment continues in Q1')
        #保存的结果路径和命名
        fileSaveName = 'Genetic_Q1_B{0}_N{1}_BU{2}_L{3}_AN{4}_E{5}'.format(blockNum,nodeNum,blockBackups,storageLimit,alloNum,epochNum)
        blockLoadQ1 = np.load('./Data/blockInfoPick_B{0}_N{1}_BU{2}_L{3}.npy'.format(blockNum,nodeNum,blockBackups,storageLimitLoad))
        blockInfo = blockInfo[np.where(blockLoadQ1==False)[0],:] #选择Q1挑选的区块
        blockNum = blockInfo.shape[0] #区块数量
        blockInfo[:,0] = np.arange(blockNum)
    else:
        fileSaveName = 'Genetic_B{0}_N{1}_BU{2}_L{3}_AN{4}_E{5}'.format(blockNum,nodeNum,blockBackups,storageLimit,alloNum,epochNum)
        
    print('Allocate {0} blocks to {1} peers.'.format(blockNum,nodeNum))
    print('Block backup is {}.'.format(blockBackups))
    print('Total block storage size is {}'.format(np.sum(blockInfo[:,1])))
    print('Total node storage limit is {}'.format(np.sum(nodeLimit)))
    #节点允许的空间限制 与 区块总大小 的比值
    nodeBlockRatio = np.sum(nodeLimit)/np.sum(blockInfo[:,1])*storageLimit
    print('Ratio of node limit to block size is {}'.format(nodeBlockRatio))
    
    #验证整个系统是否能够有足够的空间存储区块
    if (nodeBlockRatio<=blockBackups*1.2):
        print("storageLimit is too small, there's not enough space")    
    else:
        print('Storage optimization target is {}.'.format(storageLimit))
        print('Population size is {}.'.format(alloNum))
        print('Number of iterations is {}.'.format(epochNum))   
        
        #所有节点需要的花销shape=(blockNum,nodeNum,nodeNum)
        costAll = (blockInfo[:,1]*blockInfo[:,2]).reshape((blockNum,1,1)) * nodeOptDis.reshape((1,nodeNum,nodeNum))
        
        '''遗传算法'''
        bestBlockAllo = np.zeros((nodeNum,blockNum)).astype(bool) #目前最优分配
        alloEpoch = np.zeros((epochNum+1,3)) #每轮训练的最优结果 总目标 通信成本 存储平衡度 
        
        #原始种群
        #生成时单个区块数量上限(包含) 生成种群个体数量
        blockAlloGen = generate_blockAllo_backups(blockBackups+1, alloNum) #生成原始种群
         
        
        # print('Original population maximum',"%.3f"%alloTarget[np.argmax(alloFit),0])
        # print('Original population minimum',"%.3f"%alloEpoch[0,0])
        
        #存储使用率的各项结果对比
        storageVar,storageProVar = np.zeros((epochNum)),np.zeros((epochNum))
        nodeGreStorLim = np.zeros((epochNum))
        nodeGreStorLimSum,nodeGreStorLimSquaSum = np.zeros((epochNum)),np.zeros((epochNum))
        nodeGreStorLimNum = np.zeros((epochNum)).astype(int)
        nodeGreStorLimNumLayer = np.zeros((epochNum,9)).astype(int)
        
        #迭代
        for epoch in range(epochNum):
            alloFit,alloTarget = cal_fitness(blockAlloGen) #计算种群的 适应度 总目标 通信成本 存储平衡度
            alloMinIndex = np.argmin(alloTarget[:,0])
            alloEpoch[epoch,:] = alloTarget[alloMinIndex,:] # 总目标 通信成本 存储平衡度
            bestBlockAllo = blockAlloGen[alloMinIndex,:,:] #这一轮的最优分配
            
            if(epoch%20==0):
                print('Epoch',epoch)
                print('Current Best Goal',"%.3f"%alloEpoch[epoch,0])
                print('Current Max Goal',"%.3f"%alloTarget[np.argmax(alloTarget[:,0]),0])
                nodePercent = np.dot(bestBlockAllo,blockInfo[:,1])/nodeLimit
                # print('Node storage usage',nodePercent)
            
            #选择个体 进行交叉变异
            idx = select(alloFit, selectSize=alloNum) 
            blockAlloGen,alloFit,alloTarget = blockAlloGen[idx,:,:],alloFit[idx],alloTarget[idx,:]
            #交叉变异 生成下一轮的个体
            blockAlloGen = crossover_mutation(blockAlloGen, CROSSOVER_RATE=0.8)
            
            
            '''关于存储使用率的参数'''
            nodeRatio = np.dot(bestBlockAllo,blockInfo[:,1])/nodeLimit #节点存储空间占总空间的比例
            nodeProportion = (np.exp(5*nodeRatio)-1)/(np.exp(5)-1) #按照指定函数对空间占用率进行处理
            storageVar[epoch] = np.var(nodeRatio) #存储使用率的方差
            storageProVar[epoch] = np.var(nodeProportion) #存储平衡度的方差
            #超过限定值的存储状态
            nodeGreLim = nodeRatio-storageLimit #所有节点存储占用与限定值的差距
            nodeGreStorLim = nodeRatio[np.where(nodeGreLim>0)] #节点存储占用大于限定值
            nodeGreStorLimSum[epoch] = np.sum(nodeGreStorLim) #超过限定值的总和
            nodeGreStorLimSquaSum[epoch] = np.sum(np.square(nodeGreStorLim)) #超过限定值的平方和
            nodeGreStorLimNum[epoch] = len(nodeGreStorLim) #超过限定值的节点数
            nodeCnt = 0
            for index in range(9):
                nodeCnt2 = np.count_nonzero(nodeRatio<(index*0.1+0.1))
                nodeGreStorLimNumLayer[epoch,index] = nodeCnt2-nodeCnt
                nodeCnt = nodeCnt2
        
        #最后一轮
        alloFit,alloTarget = cal_fitness(blockAlloGen) #计算种群的 适应度 总目标 通信成本 存储平衡度
        alloMinIndex = np.argmin(alloTarget[:,0])
        alloEpoch[epochNum,:] = alloTarget[alloMinIndex,:] # 总目标 通信成本 存储平衡度
        bestBlockAllo = blockAlloGen[alloMinIndex,:,:] #这一轮的最优分配     
        print('Epoch',epochNum)
        print('Final Best Goal',"%.3f"%alloEpoch[epochNum,0])
        print('Final Max Goal',"%.3f"%alloTarget[np.argmax(alloTarget[:,0]),0])
        nodePercent = np.dot(bestBlockAllo,blockInfo[:,1])/nodeLimit
        
        
        #结果
        fig = plt.figure(figsize=(6,4), dpi= 300)
        plt.plot(range(epochNum+1), alloEpoch[:,0],'-',label='target value')
        # plt.plot(range(epochNum+1), alloEpoch[:,1],'-',label='communication cost')
        # plt.plot(range(epochNum+1), alloEpoch[:,2],'-',label='node storage proportion')
        plt.legend()
        # fig2 = plt.figure(figsize=(6,4), dpi= 300)
        # plt.plot(range(epochNum+1), alloEpoch[:,3],'-',label='node storage ratio variance')
        
        
        # config = {
        # "font.family":'serif',
        # #"font.size": 18,
        # "mathtext.fontset":'stix',
        # "font.serif": ['SimSun'],
        # }
        # rcParams.update(config)
        # plt.xlabel('迭代次数')
        # plt.ylabel('优化目标值')
        
        if(saveResult):
            np.savez('./Result/Q2/'+fileSaveName,alloEpoch) #优化目标
            array_dict = {'storageVar':storageVar,'storageProVar':storageProVar,
                          'nodeGreStorLimSum':nodeGreStorLimSum,'nodeGreStorLimSquaSum':nodeGreStorLimSquaSum,
                          'nodeGreStorLimNum':nodeGreStorLimNum,'nodeGreStorLimNumLayer':nodeGreStorLimNumLayer}
            np.savez('./Result/Q2/'+fileSaveName+'_Storage',**array_dict) #优化目标 
            fig.savefig('./IterationChart/Q2/'+fileSaveName+'.svg',dpi=300,format='svg',bbox_inches = 'tight')
        