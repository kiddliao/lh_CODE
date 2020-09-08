def DIJ_compute(matrix, v0, N):
    '''
    visited保存每个节点是否被访问
    distance代表v0到每个节点的最短路径
    prior代表每个节点的最短路径的前驱
    '''
    visited = [False] * N
    distance = [MaxInt] * N
    prior = [-1] * N
    for i in range(N):
        if i == v0:
            continue
        if matrix[v0][i] < MaxInt:
            distance[i] = matrix[v0][i]
            prior[i] = v0
    visited[v0] = True
    distance[v0] = 0
    # print(visited,distance,prior,sep='\n')
    #执行n-1次循环
    for _ in range(N - 1):
        curMin = MaxInt
        for w in range(N):
            if visited[w] is False and distance[w] < curMin:
                v, curMin = w, distance[w]
        visited[v] = True
        for w in range(N):
            if visited[w] is False and distance[w] > distance[v] + matrix[v][w]:
                distance[w] = distance[v] + matrix[v][w]
                prior[w] = v
    return distance,prior


def DIJ(matrix, v0, N):
    '''
    计算节点v0到其他所有节点的最短路径长和最短路径 O(n^2) 如果计算任意点之间的路径则要执行n次DIJ算法也就是O(n^3)
    '''
    res, prior = DIJ_compute(matrix, v0, N)
    for i in range(len(res)):
        print('{}到{}的最短距离为{}'.format(v0, i, res[i]))
        if i == v0:
            print('最短路径为{}\n'.format('-->'.join([str(v0), str(v0)])))
            continue
        if res[i] < MaxInt:
            tmp = []
            priorNode = i
            while priorNode != v0:
                tmp.append(priorNode)
                priorNode = prior[priorNode]
            tmp.append(v0)
            tmp = tmp[::-1]
            print('最短路径为{}\n'.format('-->'.join(list(map(str, tmp)))))
        else:
            print('无最短路径\n')

def Floyd(matrix, N):
    '''
    计算任意节点之间的最短路径长和最短路径 O(n^3)
    prior是前驱
    '''
    prior = [[-1] * N for _ in range(N)]
    dp = [[MaxInt] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            dp[i][j] = matrix[i][j]
            if matrix[i][j] < MaxInt:
                prior[i][j] = i
    for k in range(N):
        for i in range(N):
            for j in range(N):
                #如果从i经k到j的路径比直接从i到j的路径更短
                if dp[i][k] + dp[k][j] < dp[i][j]:
                    dp[i][j] = dp[i][k] + dp[k][j]
                    #在i到j的路径中,将j的前驱改成k
                    prior[i][j] = prior[k][j]
    for i in range(N):
        for j in range(N):
            if i == j:
                print('{}到{}的最短距离为{}'.format(i, j, 0))
                print('最短路径为{}\n'.format('-->'.join([str(i), str(j)])))
                continue
            else:
                print('{}到{}的最短距离为{}'.format(i, j, dp[i][j]))
                if dp[i][j] < MaxInt:
                    tmp = []
                    priorNode = j
                    while priorNode != i:
                        tmp.append(priorNode)
                        priorNode = prior[i][priorNode]
                    tmp.append(i)
                    tmp = tmp[::-1]
                    print('最短路径为{}\n'.format('-->'.join(list(map(str, tmp)))))
                else:
                    print('无最短路径\n')
    for i in dp:
        print(i)
    for i in prior:
        print(i)

if __name__ == '__main__':
    MaxInt = float('inf')
    matrix1 = [[MaxInt, MaxInt, 10, MaxInt, 30, 100],
              [MaxInt, MaxInt, 5, MaxInt, MaxInt, MaxInt],
              [MaxInt, MaxInt, MaxInt, 50, MaxInt, MaxInt],
              [MaxInt, MaxInt, MaxInt, MaxInt, MaxInt, 10],
              [MaxInt, MaxInt, MaxInt, 20, MaxInt, 60],
              [MaxInt, MaxInt, MaxInt, MaxInt, MaxInt, MaxInt]]
    matrix2 = [[0, 1, MaxInt, 4],
               [MaxInt, 0, 9, 2],
               [3, 5, 0, 8],
               [MaxInt, MaxInt, 6, 0]]
               
    N = len(matrix1)
    v0 = 0
    DIJ(matrix1, v0, N)
    Floyd(matrix1,N)

