import numpy as np
import random
import math
import matplotlib.pyplot as plt

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

def initial_solution():
    blockAllo = np.zeros((nodeNum,blockNum)).astype(bool)

    while(np.all(constraint(blockAllo))==False):
        blockAllo = np.zeros((nodeNum,blockNum)).astype(bool)
        for block in range(blockNum):
            nodeChoice = random.sample(range(nodeNum), np.random.randint(blockBackups,blockBackups+1))
            blockAllo[nodeChoice,block] = True
    return blockAllo

def cost_function(state):
    nodeBlockCost = np.zeros((nodeNum,blockNum)) #节点访问区块的代价,行代表节点,列代表区块
    for node in range(nodeNum):
        for block in range(blockNum):
            nodeBlockCost[node,block] = np.min(costAll[block,state[:,block]==1,node]) #选出通信成本最小的节点所需成本
    nodeRatio = np.dot(state,blockInfo[:,1])/nodeLimit #节点存储空间占总空间的比例
    #nodeVar = np.var(nodeRatio)
    nodeProportion = (np.exp(5*nodeRatio)-1)/(np.exp(5)-1) #按照指定函数对空间占用率进行处理
    
    #分别为 通信成本 存储平衡度 总目标
    #计算总目标时乘以 1/区块数量
    nodeBlockCostAvg = np.sum(nodeBlockCost)/(nodeNum*blockNum)
    nodeProportionAvg = np.sum(nodeProportion)/nodeNum*50
    # result = np.array([nodeBlockCostAvg+nodeProportionAvg, nodeBlockCostAvg, nodeProportionAvg])
    return nodeBlockCostAvg + nodeProportionAvg

def objective(state):
    nodeBlockCost = np.zeros((nodeNum,blockNum)) #节点访问区块的代价,行代表节点,列代表区块
    for node in range(nodeNum):
        for block in range(blockNum):
            nodeBlockCost[node,block] = np.min(costAll[block,state[:,block]==1,node]) #选出通信成本最小的节点所需成本
    nodeRatio = np.dot(state,blockInfo[:,1])/nodeLimit #节点存储空间占总空间的比例
    #nodeVar = np.var(nodeRatio)
    nodeProportion = (np.exp(5*nodeRatio)-1)/(np.exp(5)-1) #按照指定函数对空间占用率进行处理
    
    #分别为 通信成本 存储平衡度 总目标
    #计算总目标时乘以 1/区块数量
    nodeBlockCostAvg = np.sum(nodeBlockCost)/(nodeNum*blockNum)
    nodeProportionAvg = np.sum(nodeProportion)/nodeNum*50
    result = np.array([nodeBlockCostAvg+nodeProportionAvg, nodeBlockCostAvg, nodeProportionAvg])
    return result

def simulated_annealing(initial_temp=0.1, cooling_rate=0.99999, min_temp=0.001, max_iterations=100):

    current_solution = initial_solution()
    current_cost = cost_function(current_solution)

    best_solution = current_solution
    best_cost = current_cost

    temp = initial_temp
    i = 0

    while temp > min_temp and i < epochNum:
        for _ in range(max_iterations):
            # 生成新解
            random_block_num = np.random.randint(0, blockNum)
            new_solution = current_solution.copy()
            new_solution[:,random_block_num] = 0
            idx = np.random.choice(np.arange(new_solution.shape[0]) ,size = np.random.randint(blockBackups,blockBackups+1), replace=False)
            new_solution[idx,random_block_num] = 1

            while(np.all(constraint(new_solution)) == False):
                new_solution[:,random_block_num] = 0    
                random_block_num = np.random.randint(0, blockNum)
                idx = np.random.choice(np.arange(new_solution.shape[0]) ,size = np.random.randint(blockBackups,blockBackups+1), replace=False)
                new_solution[:,random_block_num] = 0
                new_solution[idx, random_block_num] = 1

            new_cost = cost_function(new_solution)

            # print(math.exp((current_cost - new_cost) / temp))
            # print(random.random())
            # print("111")

            if new_cost < current_cost or math.exp((current_cost - new_cost) / temp) > random.random():
                current_solution = new_solution
                current_cost = new_cost

                if new_cost < best_cost:
                    best_solution = new_solution
                    best_cost = new_cost
        i+=1
        r = objective(best_solution) 
        print(i,r)
        alloEpoch[i] = r
        temp *= cooling_rate

    return best_solution


if __name__=="__main__":
    blockNum = 100 #区块数量
    nodeNum = 30 #节点数量
    blockBackups = 4 #区块备份数量 至少为1
    storageLimit = 0.65 #优化后系统总存储空间占比
    storageLimitLoad = 0.5 #从Q1加载的限制区块数据量
    saveResult = True #是否保存数据 True False
    expfromQ1 = False

    INITIAL_TEMPERATURE = 0.1  # 初始温度
    FINAL_TEMPERATURE = 0.00000001  # 最终温度
    COOLING_RATE = 0.9  # 降温速率
    epochNum = 100 #迭代次数

    #区块信息
    blockInfo = np.loadtxt('./Data/blockInfo{}.csv'.format(blockNum), delimiter=',')
    
    #节点信息
    nodeLimit = np.loadtxt('./Data/NodeInfo/nodeLimit-{}.csv'.format(nodeNum), delimiter=',')
    nodeOptDis = np.load('./Data/NodeInfo/nodeRouteInfo-{}-30-100-100.npz'.format(nodeNum))['arr_0']
    nodeStorage = np.sum(nodeLimit) #CU本地存储的空间和
    storageLimitcondition = np.sum(nodeLimit)*storageLimit #CU所有节点的空间限制    

    print('Start...') 

    fileSaveName = 'CA_B{0}_N{1}_BU{2}_L{3}_E{4}'.format(blockNum,nodeNum,blockBackups,storageLimit,epochNum)

    print('Allocate {0} blocks to {1} peers.'.format(blockNum,nodeNum))
    print('Block backup is {}.'.format(blockBackups))
    print('Total block storage size is {}'.format(np.sum(blockInfo[:,1])))
    print('Total node storage limit is {}'.format(np.sum(nodeLimit)))
    #节点允许的空间限制 与 区块总大小 的比值
    nodeBlockRatio = np.sum(nodeLimit)/np.sum(blockInfo[:,1])*storageLimit
    print('Ratio of node limit to block size is {}'.format(nodeBlockRatio))

    if (nodeBlockRatio<=blockBackups*1.2):
        print("storageLimit is too small, there's not enough space")    
    else:
        alloEpoch = np.zeros((epochNum+1,3)) #每轮训练的最优结果 总目标 通信成本 存储平衡度 
        print('Storage optimization target is {}.'.format(storageLimit))
        print(f'Initial Temperature {INITIAL_TEMPERATURE}')
        print(f'Final Temperature {FINAL_TEMPERATURE}')
        print(f'Cooling Rate {COOLING_RATE}')
            #所有节点需要的花销shape=(blockNum,nodeNum,nodeNum)

        costAll = (blockInfo[:,1]*blockInfo[:,2]).reshape((blockNum,1,1)) * nodeOptDis.reshape((1,nodeNum,nodeNum))
        '''退火算法'''

        # 使用模拟退火算法搜索最优状态
        final_state = simulated_annealing()
        print("Final state: ", final_state)
        # 计算最优状态的能量
        final_energy = cost_function(final_state)
        print("Final energy: ", final_energy)

        fig = plt.figure(figsize=(6,4), dpi= 300)
        plt.plot(range(epochNum+1), alloEpoch[:,0],'-',label='target value')
        # plt.plot(range(n_iterations+1), alloEpoch[:,1],'-',label='communication cost')
        # plt.plot(range(n_iterations+1), alloEpoch[:,2],'-',label='node storage proportion')
        plt.legend()
    
    if saveResult:
        np.save('./Data/Result/Q2/{}'.format(fileSaveName),final_state)
        fig.savefig('./IterationChart/Q2/'+fileSaveName+'.svg',dpi=300,format='svg',bbox_inches = 'tight')


