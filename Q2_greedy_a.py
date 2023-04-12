import numpy as np


import numpy as np

def greedy_algorithm(storage_capacity, block_sizes, comm_costs, num_copies=4):
    num_blocks = len(block_sizes)
    num_nodes = len(storage_capacity)

    # 初始化分配策略
    blockAllo = np.zeros((nodeNum,blockNum)).astype(bool)

    # 获取区块的通信成本排序
    block_comm_cost_ranking = np.argsort(np.sum(np.sum(comm_costs, axis=1), axis=1))

    for block_idx in block_comm_cost_ranking:
        copies_assigned = 0
        while copies_assigned < num_copies:
            min_cost = float('inf')
            selected_node = None
            for node_idx in range(num_nodes):
                if storage_capacity[node_idx] * 0.0225 >= block_sizes[block_idx] and blockAllo[node_idx][block_idx] == False:
                    cost = np.sum([comm_costs[block_idx][node_idx][other_node_idx] for other_node_idx in range(num_nodes) if other_node_idx != node_idx])
                    if cost < min_cost:
                        min_cost = cost
                        selected_node = node_idx

            if selected_node is not None:
                blockAllo[selected_node,block_idx] = True
                storage_capacity[selected_node] -= block_sizes[block_idx]
                copies_assigned += 1
            else:
                raise ValueError("无法分配所有区块")

    return blockAllo


def calculate_target(alloc):
    nodeBlockCost = np.zeros((nodeNum,blockNum)) #节点访问区块的代价,行代表节点,列代表区块
    for node in range(nodeNum):
        for block in range(blockNum):
            nodeBlockCost[node,block] = np.min(costAll[block,alloc[:,block]==1,node]) #选出通信成本最小的节点所需成本
    nodeRatio = np.dot(alloc,blockInfo[:,1])/nodeLimit #节点存储空间占总空间的比例
    print(nodeRatio)
    #nodeVar = np.var(nodeRatio)
    nodeProportion = (np.exp(5*nodeRatio)-1)/(np.exp(5)-1) #按照指定函数对空间占用率进行处理
    
    #分别为 通信成本 存储平衡度 总目标
    #计算总目标时乘以 1/区块数量
    nodeBlockCostAvg = np.sum(nodeBlockCost)/(nodeNum*blockNum)
    nodeProportionAvg = np.sum(nodeProportion)/nodeNum*50
    # result = np.array([nodeBlockCostAvg+nodeProportionAvg, nodeBlockCostAvg, nodeProportionAvg])
    return nodeBlockCostAvg + nodeProportionAvg

if __name__ == "__main__":
    '''初始输入信息'''
    blockNum = 100 #区块数量
    nodeNum = 30 #节点数量
    blockBackups = 4 #区块备份数量 至少为1
    storageLimit = 0.65 #优化后系统总存储空间占比
    storageLimitLoad = 0.5 #从Q1加载的限制区块数据量
    alloNum = 10 #种群数量
    epochNum = 10000 #迭代次数
    saveResult = True #是否保存数据 True False
    expfromQ1 = False

    #区块信息
    blockInfo = np.loadtxt('./Data/blockInfo{}.csv'.format(blockNum), delimiter=',')
    
    #节点信息
    nodeLimit = np.loadtxt('./Data/NodeInfo/nodeLimit-{}.csv'.format(nodeNum), delimiter=',')
    nodeOptDis = np.load('./Data/NodeInfo/nodeRouteInfo-{}-30-100-100.npz'.format(nodeNum))['arr_0']
    nodeStorage = np.sum(nodeLimit) #CU本地存储的空间和
    storageLimitcondition = np.sum(nodeLimit)*storageLimit #CU所有节点的空间限制

    fileSaveName = 'Greedy_B{0}_N{1}_BU{2}_L{3}_AN{4}_E{5}'.format(blockNum,nodeNum,blockBackups,storageLimit,alloNum,epochNum)

    costAll = (blockInfo[:,1]*blockInfo[:,2]).reshape((blockNum,1,1)) * nodeOptDis.reshape((1,nodeNum,nodeNum))
    # print(costAll)

    np.set_printoptions(threshold=np.inf)
    allocation = greedy_algorithm(nodeLimit, blockInfo[:,1], costAll)
    print(allocation)

    print(calculate_target(allocation))
    
