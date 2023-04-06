# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 22:14:32 2022
Q1_GA
@author: Lethe
"""
import numpy as np
import matplotlib.pyplot as plt
import random

np.random.seed(1037)
random.seed(1037)

'''计算优化目标'''
def cal_objective(blockAllo,blockInfo):
    # result = np.sum((blockInfo[:,2]+blockInfo[:,3])*blockAllo) #优化目标计算结果
    #更复杂的一种约束
    u = 0.3 #表示存储于云的成本:存储于本地节点 比例参数
    #0.05是存储指标和访问指标的比例参数,否则会导致访问指标不起作用
    result = 0.05*(np.sum(blockInfo[:,1])-(1-u)*np.sum(blockInfo[:,1]*blockAllo)) + np.sum((blockInfo[:,2]+blockInfo[:,3])*blockAllo)
    return result

'''计算某种分配下CU的存储空间占用情况'''
def cal_storage(blockAllo,blockInfo):
    storage = np.sum((1-blockAllo)*blockInfo[:,1]) #该分配下的存储空间占用
    return storage

'''判定是否满足约束条件'''
def constraint(blockAllo):
    storage = cal_storage(blockAllo,blockInfo) #该分配下的存储空间占用
    return storage <= condition

'''随机生成区块分配 1代表存储于云 按照可以存储的区块数量生成'''
def generate_blockAllo(blockNum,conditionBlockNum,alloNum=10):
    blockAlloGen = np.zeros((alloNum,blockNum)).astype(bool)
    for i in range(alloNum):
        blockAllo = np.zeros(blockNum).astype(bool)
        while(constraint(blockAllo)==False):
            blockAllo = np.zeros(blockNum).astype(bool)
            idx = np.random.choice(np.arange(blockNum), size=conditionBlockNum, replace=False)
            blockAllo[idx] = True
        blockAlloGen[i,:] = blockAllo
        
    return blockAlloGen

'''计算个体的适应度'''
def cal_fitness(blockAlloGen,blockInfo):
    alloNum = blockAlloGen.shape[0]
    #行代表一种分配/个体 列依次为 适应度 存储占用 目标
    alloFit = np.zeros((alloNum,3)) 
    for index,blockAllo in enumerate(blockAlloGen): #遍历
        alloFit[index,1] = cal_storage(blockAllo,blockInfo) 
        alloFit[index,2] = cal_objective(blockAllo,blockInfo)
         
    alloFit[:,0] = -(alloFit[:,2]-np.max(alloFit[:,2]))+1e-5 #计算适应度 适应度为正数
    #alloFit[:,0] = np.exp(alloFit[:,0]*10)
    return alloFit

'''选择一定数量的个体'''
def select(blockAlloGen, alloFit, selectSize=1):
    idx = np.random.choice(np.arange(alloFit.shape[0]), size=selectSize, replace=False, #replace代表是否能重复抽取
                           p=(alloFit[:,0])/(alloFit[:,0].sum()) )
    return blockAlloGen[idx,:],alloFit[idx,:]

'''交叉变异'''
def crossover_mutation(blockAlloGen, CROSSOVER_RATE=0.8, MUTATION_RATE=0.1):
    blockAlloNew = []
    alloSize = blockAlloGen.shape[0]
    blockNum = blockAlloGen.shape[1]
    crossover_flag, mutation_flag = 0,0
    for alloFa in blockAlloGen:		#遍历种群中的每一个个体，将该个体作为父亲
        blockAlloNew.append(alloFa)
        alloChild = alloFa.copy()		#孩子先得到父亲的全部基因
        #交叉
        if np.random.rand() < CROSSOVER_RATE:			#以一定的概率发生交叉
            crossover_flag = 1
            alloMa = blockAlloGen[np.random.randint(alloSize),:]	#再种群中选择另一个个体，并将该个体作为母亲
            alloChild2 = alloMa.copy()
            crossPoints = np.random.randint(low=0, high=blockNum)	#随机产生交叉的点
            alloChild[crossPoints:] = alloMa[crossPoints:] #孩子得到位于交叉点后母亲的基因 列 
            alloChild2[:crossPoints] = alloFa[:crossPoints]
        #变异
        if np.random.rand() < MUTATION_RATE: 				#以MUTATION_RATE的概率进行变异
            mutation_flag = 1
            mutate_block = np.random.randint(0, blockNum)	#随机产生一个实数 代表要变异基因的位置
            alloChild[mutate_block] = alloChild[mutate_block]^1 	#将变异点的二进制为反转

        if(crossover_flag|mutation_flag):
            if constraint(alloChild):
                blockAlloNew.append(alloChild)
            #孩子2    
            if(crossover_flag):    
                if constraint(alloChild2):
                    blockAlloNew.append(alloChild2)   
            crossover_flag, mutation_flag = 0,0

    return np.array(blockAlloNew)

if __name__=="__main__":
    '''初始输入信息'''
    blockNum = 10 #区块数量
    nodeNum = 3 #节点数量
    storageLimit = 0.5 #优化后系统总存储空间占比
    alloNum = 10 #种群数量
    epochNum = 1000 #迭代次数
    blockBackups = 3 #区块备份数量 至少为1
    saveResult = True #是否保存数据
    

    #区块和节点信息
    blockInfo = np.loadtxt('./Data/blockInfo{}.csv'.format(blockNum), delimiter=',')
    nodeInfo = np.loadtxt('./Data/nodeInfo{}.csv'.format(nodeNum), delimiter=',')
    
    nodeStorage = np.sum(nodeInfo[:,1]) #CU本地存储的空间和
    condition = nodeStorage*storageLimit/blockBackups #优化条件/最大空间
    conditionBlockNum = int(blockNum-condition/8+blockNum*0.02) #上传到云的初始化区块数量 置1的区块数量
    
    print('Start...')
    print('Allocate {0} blocks to cloud and CU.'.format(blockNum))
    print('CU Storage space is {}.'.format(nodeStorage))
    print('The storage space target is {}.'.format(condition))
    print('All block space required is {}.'.format(np.sum(blockInfo[:,1])*blockBackups))
    print('Storage optimization target is {}.'.format(storageLimit))
    print('Population size is {}.'.format(alloNum))
    print('Number of iterations is {}.'.format(epochNum))
    
    '''遗传算法信息'''
    bestBlockAllo = np.zeros(blockNum).astype(bool) #目前最优分配
    alloFitEpoch = np.zeros((epochNum+1,2)) #每轮训练的最优结果 通信成本 存储平衡度 总目标
    
    #原始种群
    blockAlloGen = generate_blockAllo(blockNum,conditionBlockNum,alloNum) #生成原始种群
    alloFit = cal_fitness(blockAlloGen,blockInfo) #计算原始种群的 总目标 适应度
    alloFitEpoch[0,:] = alloFit[np.argmin(alloFit[:,2]),1:]
    print('Original population maximum',alloFit[np.argmax(alloFit[:,2]),-1])
    print('Original population minimum',alloFitEpoch[0,-1])
    #迭代
    for epoch in range(epochNum):
        epochNow = epoch+1
        blockAlloGen = crossover_mutation(blockAlloGen, CROSSOVER_RATE=0.8) #交叉变异
        #print(blockAlloGen.shape[0])
        alloFit = cal_fitness(blockAlloGen,blockInfo) #计算种群适应度
        blockAlloGen,alloFit = select(blockAlloGen, alloFit, selectSize=alloNum) #选择个体
        alloFitEpoch[epochNow,:] = alloFit[np.argmin(alloFit[:,2]),1:] #这一轮的最优值
        # print('Current Best',alloFitEpoch[epoch+1,:])
        bestBlockAllo = blockAlloGen[np.argmin(alloFit[:,2]),:] #这一轮的最优分配
        
        if(epochNow%50==0):
            print('Epoch',epochNow)
            print('Current Best Goal',alloFitEpoch[epochNow,-1])    
    
    #结果
    fig = plt.figure(figsize=(6,4), dpi= 300)
    plt.plot(range(epochNum+1), alloFitEpoch[:,1],'-')
    if(saveResult):
        np.save('./Result/Q1/Genetic_B{0}_N{1}_BU{2}_L{3}_AN{4}_E{5}'.format(blockNum,nodeNum,blockBackups,storageLimit,alloNum,epochNum),
                alloFitEpoch)
        # blockSave = blockInfo[np.where(bestBlockAllo==False)[0],:]
        # blockSave[:,0] = np.arange(blockSave.shape[0])
        np.save('./Data/blockInfoPick_B{0}_N{1}_BU{2}_L{3}'.format(blockNum,nodeNum,blockBackups,storageLimit), 
                bestBlockAllo)
        fig.savefig('./IterationChart/Q1/Genetic_B{0}_N{1}_BU{2}_L{3}_AN{4}_E{5}.svg'.format(blockNum,nodeNum,blockBackups,storageLimit,alloNum,epochNum),
                    dpi=300,format='svg',bbox_inches = 'tight')
    







