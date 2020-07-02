from collections import defaultdict
class Solution:
    def longestIncreasingPath(self, matrix) -> int:
        if not matrix:return 0
        m,n=len(matrix),len(matrix[0])
        dp=[[0]*n for i in range(m)]#计算以i,j开始的最长路径
        def backtrack1(i,j,res):
            index=[]
            tmp1,tmp2=0,res#tmp1是四周已经算过的增量 tmp2是没算过的直接去回溯
            for x,y in [(i+1,j),(i-1,j),(i,j+1),(i,j-1)]:
                if 0<=x<m and 0<=y<n and matrix[i][j]<matrix[x][y]:
                    if dp[x][y]!=0:
                        tmp1=max(tmp1,dp[x][y])
                    else:
                        index.append((x,y))
            for x,y in index:
                save=matrix[i][j]
                matrix[i][j]=-float('inf')
                tmp2 = max(tmp2,backtrack(x, y, res + 1))
                matrix[i][j]=save
            return max(tmp1+res,tmp2)           
            
        def backtrack2(i, j, res):#深度遍历非递归 遍历是可以遍历 但是不好确定最大深度 这个没做出来
            stack = [(i,j)]
            res = 1
            visited = set()
            ans=[]
            while stack:
                x, y = stack.pop()
                ans.append((x, y))
                res=max(res,len(ans))
                count=0
                for a, b in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]:
                    if 0 <= a < len(matrix) and 0 <= b < len(matrix[0]) and matrix[x][y] < matrix[a][b]:
                        if dp[a][b] != 0:
                            res = max(len(ans) + dp[a][b], res)
                        else:
                            stack.append((a, b))
                            #要是一个都没入栈说明这条路走到头
                            count += 1
                if count==0:ans.pop()
            return max(res, count)

        def index3(i, j):
            res = []
            for x, y in [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)]:
                if 0 <= x < len(matrix) and 0 <= y < len(matrix[0]) and matrix[i][j] < matrix[x][y]:
                    res.append((x, y))
            return res

        def backtrack3(i, j, res):  #还是深度遍历的非递归 模仿前序遍历的写法 没搞定 二叉树的很简单 多叉树和图有难度
            stack = []
            p = (i, j)
            child = defaultdict(list)
            res=1
            while stack and index3(*p):
                while index3(*p):
                    stack.append(p)
                    res=max(res,len(stack))
                    child[(p[0], p[1])] += index3(*p)
                    p = child[(p[0], p[1])].pop(0)  #第一个儿子先出栈
                tmp = stack.pop()
                p = tmp
            return res
                    

            
                
        ans=1
        for i in range(m):
            for j in range(n):
                tmp=backtrack2(i,j,1)
                # if tmp>=m*n:return tmp
                dp[i][j]=tmp
                ans = max(ans, tmp)
        for i in dp:print(i)
        return ans
                    
                        
                        

                        
                        
a = Solution()
print(a.longestIncreasingPath([[1,2],[2,3]]))
print(a.longestIncreasingPath([[7,6,1,1],[2,7,6,0],[1,3,5,1],[6,6,3,2]]))
print(a.longestIncreasingPath([[0,1,2,3,4,5,6,7,8,9],[19,18,17,16,15,14,13,12,11,10],[20,21,22,23,24,25,26,27,28,29],[39,38,37,36,35,34,33,32,31,30],[40,41,42,43,44,45,46,47,48,49],[59,58,57,56,55,54,53,52,51,50],[60,61,62,63,64,65,66,67,68,69],[79,78,77,76,75,74,73,72,71,70],[80,81,82,83,84,85,86,87,88,89],[99,98,97,96,95,94,93,92,91,90],[100,101,102,103,104,105,106,107,108,109],[119,118,117,116,115,114,113,112,111,110],[120,121,122,123,124,125,126,127,128,129],[139,138,137,136,135,134,133,132,131,130],[0,0,0,0,0,0,0,0,0,0]] ))