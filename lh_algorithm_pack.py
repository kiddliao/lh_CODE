#一.0-1背包
#n个物体,每个物体只有一件,背包容量为v,wi是第i个物体放入后获得的价值
#ci是第i个物体占用的背包空间
#求让背包内物体价值和做大的放法
v = 8
c = [2, 3, 4, 5]
w = [3, 4, 5, 6]

def pack1_1(v, w, c):
    '''
    dp[i][j]代表前i个物体放入后背包空间为j时的价值总和\n
    dp[3][4]代表现在只放1号和2号物体和三号物体且背包空间为4的状态
    如果放入3号物体 那么此时价值为w[3]+dp[2][4-c[3]]
    即放入3 获得3的价值w[3] 此时空间为4-c[3] 
    那么问题为此时空间为4-c[3]只能放入1号物体和2号物体 即dp[2][4-c[3]]
    如果不放入3号物体 问题为空间为4只能放入1号物体和2号物体 即dp[2][4]
    dp[i][j]=max(dp[i-1][j],dp[i-1][j-c[i]]+w[i])
    '''
    n = len(w)
    dp = [[0 for i in range(v + 1)] for i in range(n + 1)]
    # 如果要求恰好放满
    # for i in range(1, v + 1):
    #     dp[0][i] = -float('inf')
    # 当空间为0或放入前0号物体时价值都为0 即第一行和第一列都是0
    for i in range(1,n+1):
        for j in range(1,v+1):
            if j < c[i - 1]:
    # 放不下i号物体那么就是空间为j只放前i号物体的问题
                dp[i][j] = dp[i - 1][j]
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - c[i - 1]] + w[i - 1])
                
    for i in dp:print(i)
    return dp[n][v]

def pack1_2(v, w, c):
    '''
    优化空间复杂度
    '''
    n = len(w)
    dp = [0 for i in range(v + 1)]
    # 如果要求恰好放满
    # for i in range(1, v + 1):
    #     dp[i] = -float('inf')
    print(dp)
    for i in range(1, n + 1):
        # 逆序
        # for j in range(v , 0, -1):
        # 常数优化
        # for j in range(v, c[i - 1] - 1, -1):
        # 常数优化
        for j in range(v,max(v-sum(c),c[i-1])-1,-1):
            if j >= c[i - 1]:
                dp[j] = max(dp[j], dp[j - c[i - 1]] + w[i - 1])
        print(dp)
    return dp[-1]
    

# print(pack1_1(v,w,c))
# print(pack1_2(v,w,c))



#二完全背包问题
#n个物体,每个物体有无限件,背包容量为v,wi是第i个物体放入后获得的价值
#ci是第i个物体占用的背包空间
#求让背包内物体价值和做大的放法
v = 8
c = [2, 3, 4, 5]
w = [3, 4, 5, 6]

def pack2_1(v, c, w):
    '''
    对于一个物体不再是只能取0次和1次
    而是可以取0,1...floor(v/c[i])次 记为k次
    dp[i][j]=max(dp[i-1][v-kc[i]]+kw[i])
    '''
    n = len(w)
    dp = [[0 for i in range(v + 1)] for j in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, v + 1):
            count = list(range(j // c[i - 1] + 1))
            imax = -float('inf')
            for k in count:
                imax = max(imax, dp[i - 1][j - k * c[i - 1]] + k * w[i - 1])
            dp[i][j] = imax

    for i in dp:print(i)
    return dp[n][v]
    
def pack2_2(v, c, w):
    '''
    有一个简单的优化
    如果有2个物体i,j c[i]<c[j]且w[i]>w[j] 那么显然可以直接去掉j
    针对背包问题来说 首先去掉代价大于V的物品 根本放不进去
    然后算出代价相同的物品中谁是价格最高的 类似计数排序
    上述优化暂不实现
    将完全背包转换01背包
    每个物品最多可以放V//c[i]次 最简单的想法是把第i个物品转换为V//c[i]个物品i放到待选列表里
    但是这样时间复杂度没变
    更高效的方法是把第i种物品拆成费用为2^k*c[i]价值为2^k*w[i]的若干见物品
    k取值普遍满足2^k*c[i]<=V的非负整数
    这是二进制的思想 不管最优策略选几件第i种物品 件数写成二进制后总可以表示成
    若干个2^k件物品的和 就把每种物品拆成了O(log(V//c[i]))件物品
    '''
    n = len(w)
    for i in range(n):
        k = 1
        while 2 ** k * c[i] <= v:
            c.append(2 ** k * c[i])
            w.append(2 ** k * w[i])
            k += 1
    n = len(w)
    dp = [[0 for i in range(v + 1)] for i in range(n + 1)]
    for i in range(1,n+1):
        for j in range(1,v+1):
            if j < c[i - 1]:
                dp[i][j] = dp[i - 1][j]
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - c[i - 1]] + w[i - 1])
                
    for i in dp:print(i)
    return dp[n][v]

def pack2_3(v, c, w):
    '''一维数组'''
    n = len(w)
    dp = [0 for i in range(v + 1)]
    print(dp)
    for i in range(1, n + 1):
        # 正序
        for j in range(1, v + 1):
            if j >= c[i - 1]:
                dp[j] = max(dp[j], dp[j - c[i - 1]] + w[i - 1])
        print(dp)
    return dp[-1]

# print(pack2_1(v, c, w))
# print(pack2_2(v, c, w))
# print(pack2_3(v, c, w))

            
    

#三多重背包问题
#和完全背包问题很类似
#每个物体限制了件数
v = 12
c = [2, 3, 4, 5]
w = [3, 4, 5, 6]
o = [2, 3, 1, 2]
# c = [2, 3]
# w = [3, 4]
# o = [2, 3]

def pack3_1(v, c, w, o):
    '''
    优化前的笨方法
    '''
    n = len(w)
    dp = [[0 for i in range(v + 1)] for i in range(n + 1)]
    for i in range(1, n + 1):
        cost, value = c[i - 1], w[i - 1]
        for j in range(1, v + 1):
            if j < c[i - 1]:
                dp[i][j] = dp[i - 1][j]
            else:
                num = min(j // cost, o[i - 1])
                imax = 0
                for k in range(num + 1):
                    imax = max(imax, dp[i - 1][j - k * cost] + k * value)
                dp[i][j] = imax
    for i in dp: print(i)
    return dp[n][v]

def pack3_2(v, c, w, o):
    '''
    二进制优化
    '''
    n = len(w)
    for i in range(n):
        k = 1
        while 2 ** k * c[i] <= v and 2 ** k < o[i]:
            c.append(2 ** k * c[i])
            w.append(2 ** k * w[i])
            k += 1
    n = len(w)
    dp = [[0 for i in range(v + 1)] for i in range(n + 1)]
    for i in range(1,n+1):
        for j in range(1,v+1):
            if j < c[i - 1]:
                dp[i][j] = dp[i - 1][j]
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - c[i - 1]] + w[i - 1])
                
    for i in dp:print(i)
    return dp[n][v]


def pack3_3(v, c, w, o):
    '''
    单调队列
    '''
    n = len(w)
    dp = [[0 for i in range(v + 1)] for i in range(n + 1)]
    for i in range(1, n + 1):
        
        cost, value = c[i - 1], w[i - 1]
        window = min(v // cost, o[i - 1]) + 1
        
        for remain in range(cost):
            queue1 = [float('inf')]
            queue2 = [0]
            for k in range((v - remain) // cost + 1):
                j = remain + k * cost
                a = j // cost
                if len(queue1) > 1 and k - queue2[1] >= window:
                    queue1.pop(1)
                    queue2.pop(1)
                while queue1[-1] <= dp[i - 1][j] - a * value:
                    queue1.pop()
                    queue2.pop()
                queue1.append(dp[i - 1][j] - a * value)
                queue2.append(k)
                dp[i][j] = queue1[1] + a * value
    for i in dp: print(i)
    return dp[n][v]

def pack3_4(v, c, w, o):
    '''
    单调队列
    '''
    n = len(w)
    dp = [0 for i in range(v + 1)]
    print(dp)
    for i in range(1, n + 1):
        
        cost, value = c[i - 1], w[i - 1]
        window = min(v // cost, o[i - 1]) + 1
        for remain in range(cost):
            queue1 = [float('inf')]
            queue2 = [0]
            for k in range((v - remain) // cost + 1):
                j = remain + k * cost
                a = j // cost
                if len(queue1) > 1 and k - queue2[1] >= window:
                    queue1.pop(1)
                    queue2.pop(1)
                while queue1[-1] <= dp[j] - a * value:
                    queue1.pop()
                    queue2.pop()
                queue1.append(dp[j] - a * value)
                queue2.append(k)
                dp[j] = queue1[1] + a * value
        print(dp)
    return dp[v]




# print(pack3_1(v, c, w, o))
# print(pack3_2(v, c, w, o))

v = 12
c = [2, 3, 4, 5]
w = [3, 4, 5, 6]
o = [2, 3, 1, 2]

# print(pack3_3(v, c, w, o))
# print(pack3_4(v, c, w, o))


#4.1 01背包和多重背包
v = 12
c = [2, 3, 4, 5]
w = [3, 4, 5, 6]
o = [1, float('inf'), float('inf'), 1]

def pack4_1(v, c, w, o):
    n = len(w)
    dp = [0 for i in range(v + 1)]
    print(dp)
    for i in range(1, n + 1):
        if o[i - 1] != 1:
            for j in range(1, v + 1):
                if j >= c[i - 1]:
                    dp[j] = max(dp[j], dp[j - c[i - 1]] + w[i - 1])
        else:
            for j in range(v, c[i - 1] - 1, -1):
                if j >= c[i - 1]:
                    dp[j] = max(dp[j], dp[j - c[i - 1]] + w[i - 1])
        print(dp)
    return dp[-1]

# print(pack4_1(v, c, w, o))

#4.2 01背包 多重背包 完全背包
v = 12
c = [2, 3, 4, 5]
w = [3, 4, 5, 6]
o = [1, float('inf'), 2, 1]

def pack4_2(v, c, w, o):
    n = len(w)
    dp = [0 for i in range(v + 1)]
    print(dp)
    for i in range(1, n + 1):
        if o[i - 1] == float('inf'):
            for j in range(1, v + 1):
                if j >= c[i - 1]:
                    dp[j] = max(dp[j], dp[j - c[i - 1]] + w[i - 1])
        elif o[i - 1] == 1:
            for j in range(v, c[i - 1] - 1, -1):
                if j >= c[i - 1]:
                    dp[j] = max(dp[j], dp[j - c[i - 1]] + w[i - 1])
        else:
            cost, value = c[i - 1], w[i - 1]
            window = min(v // cost, o[i - 1]) + 1
            for remain in range(cost):
                queue1 = [float('inf')]
                queue2 = [0]
                for k in range((v - remain) // cost + 1):
                    j = remain + k * cost
                    a = j // cost
                    if len(queue1) > 1 and k - queue2[1] >= window:
                        queue1.pop(1)
                        queue2.pop(1)
                    while queue1[-1] <= dp[j] - a * value:
                        queue1.pop()
                        queue2.pop()
                    queue1.append(dp[j] - a * value)
                    queue2.append(k)
                    dp[j] = queue1[1] + a * value

        print(dp)
    return dp[-1]

# print(pack4_2(v, c, w, o))


