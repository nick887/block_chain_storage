import numpy as np
import random
import matplotlib.pyplot as plt

class Particle:
    def __init__(self, position, velocity, cost_func):
        self.position = position
        self.velocity = velocity
        self.cost_func = cost_func
        self.cost = cost_func(position)
        self.best_position = np.copy(position)
        self.best_cost = self.cost

    def update_velocity(self, global_best_position,current_iteration, max_iterations, w_max=0.9, w_min=0.4,c1=2, c2=2):
        w = w_max - (w_max - w_min) * (current_iteration / max_iterations)
        r1, r2 = random.random(), random.random()
        cognitive = c1 * r1 * (self.best_position - self.position)
        social = c2 * r2 * (global_best_position - self.position)
        self.velocity = w * self.velocity + cognitive + social

    def update_position(self, global_best_position):
        self.position = self.position + self.velocity
        self.position = 1 / (1 + np.exp(-self.position))  # 使用 sigmoid 函数将 position 转换为 0 到 1 之间的值
        self.position = (self.position > 0.5).astype(int)  # 将 position 转换为布尔值（0 或 1）
        for i in range(self.position.shape[1]):
            block_distribution = self.position[:,i]
            while np.sum(block_distribution) > 4:
                block_distribution[np.argmax(block_distribution)] = 0
        # print(self.position)
        while(np.all(constraint(self.position)) == False):
            # print("不满足约束条件")
            self.update_velocity(global_best_position)
            self.position = self.position + self.velocity
            self.position = 1 / (1 + np.exp(-self.position))  # 使用 sigmoid 函数将 position 转换为 0 到 1 之间的值
            self.position = (self.position > 0.5).astype(int)  # 将 position 转换为布尔值（0 或 1）


    def update_cost(self):
        self.cost = self.cost_func(self.position)
        if self.cost < self.best_cost:
            self.best_position = np.copy(self.position)
            self.best_cost = self.cost

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
    blockAllo = np.zeros((nodeNum,blockNum)).astype(int)

    while(np.all(constraint(blockAllo))==False):
        blockAllo = np.zeros((nodeNum,blockNum)).astype(int)
        for block in range(blockNum):
            nodeChoice = random.sample(range(nodeNum), np.random.randint(blockBackups,blockBackups+1))
            blockAllo[nodeChoice,block] = 1
    return blockAllo

def fitness(solution):
    nodeBlockCost = np.zeros((nodeNum,blockNum)) #节点访问区块的代价,行代表节点,列代表区块
    for node in range(nodeNum):
        for block in range(blockNum):
            nodeBlockCost[node,block] = np.min(costAll[block,solution[:,block]==1,node]) #选出通信成本最小的节点所需成本
    nodeRatio = np.dot(solution,blockInfo[:,1])/nodeLimit #节点存储空间占总空间的比例
    #nodeVar = np.var(nodeRatio)
    nodeProportion = (np.exp(5*nodeRatio)-1)/(np.exp(5)-1) #按照指定函数对空间占用率进行处理
    
    #分别为 通信成本 存储平衡度 总目标
    #计算总目标时乘以 1/区块数量
    nodeBlockCostAvg = np.sum(nodeBlockCost)/(nodeNum*blockNum)
    nodeProportionAvg = np.sum(nodeProportion)/nodeNum*50
    # result = np.array([nodeBlockCostAvg+nodeProportionAvg, nodeBlockCostAvg, nodeProportionAvg])
    return nodeBlockCostAvg + nodeProportionAvg

def particle_encoding(particle,num_copies=4):
    blockAllo = np.zeros((nodeNum,blockNum)).astype(bool)
    for block_idx in range(blockNum):
        nodes = np.argsort(particle[block_idx])[:num_copies]
        blockAllo[nodes][block_idx] = True
    return blockAllo

def pso(n_particles, n_iterations, fitness):
    particles = []
    for _ in range(n_particles):
        position = initial_solution()
        velocity = np.random.rand(nodeNum, blockNum) - 0.5
        particles.append(Particle(position, velocity, fitness))

    global_best_position = np.copy(particles[0].position)
    global_best_cost = particles[0].cost

    for particle in particles[1:]:
        if particle.cost < global_best_cost:
            global_best_position = np.copy(particle.position)
            global_best_cost = particle.cost

    print(global_best_cost)

    no_improvement_count = 0
    for iteration in range(n_iterations):
        for particle in particles:
            particle.update_velocity(global_best_position, iteration, n_iterations)
            particle.update_position(global_best_position)
            particle.update_cost()

            if particle.cost < global_best_cost:
                global_best_position = np.copy(particle.position)
                global_best_cost = particle.cost
                no_improvement_count=0
            else:
                no_improvement_count+=1

            if no_improvement_count >= 3:
                for particle in particles:
                    position = np.random.rand(nodeNum, blockNum)
                    position = (position > 0.5).astype(int)
                    velocity = np.random.rand(nodeNum, blockNum) - 0.5
                    particle.position = position
                    particle.velocity = velocity
                    particle.cost = fitness(position)
                no_improvement_count = 0

        print(global_best_cost)
    return global_best_position



if __name__=="__main__":
    blockNum = 100 #区块数量
    nodeNum = 30 #节点数量
    blockBackups = 4 #区块备份数量 至少为1
    storageLimit = 0.65 #优化后系统总存储空间占比
    storageLimitLoad = 0.5 #从Q1加载的限制区块数据量
    saveResult = True #是否保存数据 True False
    expfromQ1 = False
    alloNum = 10 #种群数量
    epochNum = 1000 #迭代次数

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
        fileSaveName = 'CA_B{0}_N{1}_BU{2}_L{3}_AN{4}_E{5}'.format(blockNum,nodeNum,blockBackups,storageLimit,alloNum,epochNum)

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
            #所有节点需要的花销shape=(blockNum,nodeNum,nodeNum)

        costAll = (blockInfo[:,1]*blockInfo[:,2]).reshape((blockNum,1,1)) * nodeOptDis.reshape((1,nodeNum,nodeNum))

        n_particles = 3000
        n_iterations = 10000
        final_solution = pso(n_particles=n_particles, n_iterations=n_iterations, fitness=fitness)
        print("Final state: ", final_solution)
        # 计算最优状态的能量
        final_energy = fitness(final_solution)
        print("Final energy: ", final_energy)
    
    if saveResult:
        np.save('./Data/Result/Q2/{}'.format(fileSaveName),final_solution)
        fig.savefig('./IterationChart/Q2/'+fileSaveName+'.svg',dpi=300,format='svg',bbox_inches = 'tight')

