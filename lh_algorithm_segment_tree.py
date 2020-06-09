#线段树
#区间最大值线段树
#将线段转换为线段树
#如果涉及到多叉树 首先要把多叉树转换为线段(DFS并且记录入树出树的时间,详见力扣LCP05)再转换为线段树 
maxn = 50005
max_number = 1e9


class segment_tree:
    def __init__(self, k, l, r, a, t):
        '''
        l,r是所有节点的区间,k是线段树root下标(1 层次遍历),a是真实结点,t是线段树(记录每个区间的最大值)
        '''
        self.a = a
        self.t = t
        self.l = l
        self.r = r
        self.k = k
        self.lazy = [0] * len(self.t)
        self.Node(self.l, self.r, self.k)
    
    def Node(self, l, r, k):  #递归建树
        '''
        k是线段树root下标 线段树中的叶子节点都是真实结点 非叶子节点都是虚拟节点 l,r是所有结点区间
        '''
        if l == r:
            self.t[k] = self.a[l]
            return t[k]
        else:
            m = (l + r) >> 1
            lmax = self.Node(l, m, k << 1)
            rmax = self.Node(m + 1, r, k << 1 | 1)
            self.t[k] = max(lmax, rmax)
            return self.t[k]
    def update_plot(self, p, v, l, r, k):  #更新一个节点的值 
        '''
        p是真实结点下标 v是要加上的值
        '''
        if l == r:
            self.a[p] += v #原博客有误 https://www.cnblogs.com/xenny/p/9801703.html
            self.t[k] += v
            return self.t[k]
        else:
            m = (l + r) >> 1
            if p <= m:
                newmax = self.update_plot(p, v, l, m, k << 1)
            else:
                newmax = self.update_plot(p, v, m + 1, r, k << 1 | 1)
            self.t[k] = max(self.t[k], newmax)
            return self.t[k]
    
    def query(self, L, R, l, r, k): #区间查询
        '''
        L,R是查询区间
        '''
        if L <= l <= r <= R:
            return self.t[k]
        else:
            res = 0
            m = (l + r) >> 1
            lmax_LR, rmax_LR = 0, 0
            if L <= m:
                lmax_LR = self.query(L, R, l, m, k << 1)
            if R > m:
                rmax_LR = self.query(L, R, m + 1, r, k << 1 | 1)
            res = max(lmax_LR, rmax_LR)
            return res
    
    def lazy_update_block(self, L, R, v, l, r, k):  #区间更新 比如对1-3的每个节点都加3
        '''
        如果对节点的所有子节点都进行更新操作,复杂度为O(nlogn)
        为提高效率,引入延迟更新,更新到结点区间为需要更新的区间
        的真子集不再往下更新,下次若是遇到需要用这下面的结点的
        信息,再去更新这些结点,所以这样的话使得区间更新的操作
        和区间查询类似,复杂度为O(logN)
        ''' 
        if L <= l <= r <= R:
            self.lazy[k] += v
            self.t[k] += v  #加到真子集就不再往下的子节点加
            return self.t[k]
        else:
            if self.lazy[k]:
                self.lazy[k << 1] += self.lazy[k]
                self.lazy[k << 1 | 1] += self.lazy[k]  #如果要到子节点先把上次延迟更新的更新一波
                self.t[k >> 1] += self.lazy[k]
                self.t[k >> 1 | 1] += self.lazy[k]
                self.lazy[k] = 0
                
            m = (l + r) >> 1
            lmax, rmax = 0, 0
            if L <= m:
                lmax = self.lazy_update_block(L, R, v, l, m, k << 1)
            if R > m:
                rmax = self.lazy_update_block(L, R, v, m + 1, r, k << 1 | 1)
            t[k] = max(lmax, rmax)
            return t[k]
    
    def lazy_query(self, L, R, l, r, k): #区间查询
        '''
        L,R是查询区间
        '''
        if L <= l <= r <= R:
            return self.t[k]
        else:
            if self.lazy[k]:
                self.lazy[k << 1] += self.lazy[k]
                self.lazy[k << 1 | 1] += self.lazy[k]  #如果要到子节点先把上次延迟更新的更新一波
                self.t[k >> 1] += self.lazy[k]
                self.t[k >> 1 | 1] += self.lazy[k]
                self.lazy[k] = 0

            res = 0
            m = (l + r) >> 1
            lmax_LR, rmax_LR = 0, 0
            if L <= m:
                lmax_LR = self.lazy_query(L, R, l, m, k << 1)
            if R > m:
                rmax_LR = self.lazy_query(L, R, m + 1, r, k << 1 | 1)
            res = max(lmax_LR, rmax_LR)
            return res


a = [0] * maxn #原区间
t = [0] * (maxn // 2) #线段树
a[1:7] = [1, 8, 6, 4, 3, 5]

tree = segment_tree(1, 1, 6, a, t)
print(tree.query(3, 6, tree.l, tree.r, tree.k))
print(tree.query(4, 6, tree.l, tree.r, tree.k))
tree.update_plot(3, 7, tree.l, tree.r, tree.k)
print(tree.query(3, 6, tree.l, tree.r, tree.k))
print(tree.query(4, 6, tree.l, tree.r, tree.k))

tree.lazy_update_block(1, 3, 7, tree.l, tree.r, tree.k)
print(tree.lazy_query(3, 6, tree.l, tree.r, tree.k))
print(tree.lazy_query(4, 6, tree.l, tree.r, tree.k))       

            
    
            
            


