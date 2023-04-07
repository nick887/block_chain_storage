import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib import rcParams
import math

np.random.seed(1037)
random.seed(1037)

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

'''随机生成区块 按照备份数量生成'''
def generate_initial_state(blockBackupsUpper):
    blockAllo = np.zeros((nodeNum,blockNum)).astype(bool)
    blockBackupsUpper = blockBackupsUpper+1
    #blockBackupsUpper = int(blockBackups*2)

    while(np.all(constraint(blockAllo))==False):
        blockAllo = np.zeros((nodeNum,blockNum)).astype(bool)
        for block in range(blockNum):
            nodeChoice = random.sample(range(nodeNum), np.random.randint(blockBackups,blockBackupsUpper))
            blockAllo[nodeChoice,block] = True
        #print(constraint(blockAllo))
        
    return blockAllo

# 计算状态的能量，即系统存储平衡度
def calculate_energy(state):
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

# 生成一个邻居状态，即随机选择一个区块，然后将其从一个随机节点移动到另一个随机节点上
def get_neighbour(state):
    node_usage = np.dot(state,blockInfo[:,1])/nodeLimit
    node_max = np.argmax(node_usage[:,0])
    node_min = np.argmin(node_usage[:,0])

    blockInfo[state[node_max],1]
    
    # random_block_num = np.random.randint(0, blockNum)
    # neighbour = state.copy()
    # neighbour[:,random_block_num] = 0
    
    # idx = np.random.choice(np.arange(state.shape[0]) ,size = np.random.randint(blockBackups,blockBackups+1), replace=False)
    # neighbour[idx,random_block_num] = 1
    # while(np.all(constraint(state)) == False):
    #     neighbour[:,random_block_num] = 0    
    #     random_block_num = np.random.randint(0, blockNum)
    #     idx = np.random.choice(np.arange(state.shape[0]) ,size = np.random.randint(blockBackups,blockBackups+1), replace=False)
    #     neighbour[:,random_block_num] = 0
    #     neighbour[idx, random_block_num] = 1

    #TODO add constraint
    return neighbour


# 模拟退火算法
def simulated_annealing(initial_state, initial_temperature, final_temperature, cooling_rate):
    current_state = initial_state
    current_energy = calculate_energy(current_state)
    temperature = initial_temperature
    while temperature > final_temperature:
        neighbour = get_neighbour(current_state)
        neighbour_energy = calculate_energy(neighbour)
        energy_delta = neighbour_energy - current_energy
        if energy_delta < 0:
            current_state = neighbour
            current_energy = neighbour_energy
        else:
            probability = math.exp(-energy_delta / temperature)
            if random.random() < probability:
                current_state = neighbour
                current_energy = neighbour_energy
        temperature *= cooling_rate
        print(calculate_energy(current_state))
    return current_state

if __name__=="__main__":
    blockNum = 100 #区块数量
    nodeNum = 30 #节点数量
    blockBackups = 4 #区块备份数量 至少为1
    storageLimit = 0.65 #优化后系统总存储空间占比
    storageLimitLoad = 0.5 #从Q1加载的限制区块数据量
    saveResult = True #是否保存数据 True False
    expfromQ1 = False
    INITIAL_TEMPERATURE = 100.0  # 初始温度
    FINAL_TEMPERATURE = 0.1  # 最终温度
    COOLING_RATE = 0.99  # 降温速率
    alloNum = 10 #种群数量
    epochNum = 10000 #迭代次数

    #区块信息
    blockInfo = np.loadtxt('./Data/blockInfo{}.csv'.format(blockNum), delimiter=',')
    
    #节点信息
    nodeLimit = np.loadtxt('./Data/NodeInfo/nodeLimit-{}.csv'.format(nodeNum), delimiter=',')
    nodeOptDis = np.load('./Data/NodeInfo/nodeRouteInfo-{}-30-100-100.npz'.format(nodeNum))['arr_0']
    nodeStorage = np.sum(nodeLimit) #CU本地存储的空间和
    storageLimitcondition = np.sum(nodeLimit)*storageLimit #CU所有节点的空间限制    

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

    if (nodeBlockRatio<=blockBackups*1.2):
        print("storageLimit is too small, there's not enough space")    
    else:
        print('Storage optimization target is {}.'.format(storageLimit))
        print(f'Initial Temperature {INITIAL_TEMPERATURE}')
        print(f'Final Temperature {FINAL_TEMPERATURE}')
        print(f'Cooling Rate {COOLING_RATE}')
            #所有节点需要的花销shape=(blockNum,nodeNum,nodeNum)

        costAll = (blockInfo[:,1]*blockInfo[:,2]).reshape((blockNum,1,1)) * nodeOptDis.reshape((1,nodeNum,nodeNum))
        '''退火算法'''

        # 生成初始状态
        initial_state = generate_initial_state(blockBackups+1)
        print("Initial state: ", initial_state)
        # 使用模拟退火算法搜索最优状态
        final_state = simulated_annealing(initial_state, INITIAL_TEMPERATURE, FINAL_TEMPERATURE, COOLING_RATE)
        print("Final state: ", final_state)
        # 计算最优状态的能量
        final_energy = calculate_energy(final_state)
        print("Final energy: ", final_energy)
    

