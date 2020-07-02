inf=float('inf')
#https://blog.csdn.net/qq_43571807/article/details/100941031
class Graph:
    def __init__(self,mat,vertices,flag):#flag=1是有向图，0是无向图
        self.flag=flag
        self.x=inf if self.flag else 0
        self.vertices=vertices
        self.mat=mat
        self.edges_dict={}
        self.edges_list=[]
        #如果给邻接矩阵就创建edges字典
        if len(mat)>0:
            if len(mat)!=len(vertices):
                raise IndexError
            self.edges_dict,self.edges_list=self.getAllEdges()
        #如果没给邻接矩阵只给了顶点就初始化个空mat 有向图用inf初始化,无向图用0初始化
        elif len(vertices) > 0:
            self.mat = [[x for col in range(len(vertices))] for row in range(len(vertices))]
        
        self.edges_num=len(self.edges_list)
        self.vertices_num=len(self.vertices)
        
    def getAllEdges(self):
        dic={}
        list=[]
        for i in range(len(self.mat)):
            for j in range(len(self.mat)):
                if (self.flag==0 and self.mat[i][j]>0) or (self.flag==1 and self.mat[i][j]<inf):
                    dic[(self.vertices[i],self.vertices[j])]=self.mat[i][j]
                    list.append((self.vertices[i],self.vertices[j],self.mat[i][j]))
        return dic,list
    def getAllVertices(self):
        return self.vertices
    
    def isOutRange(self, x):
        try:
            if x >= self.vertices_num or x <= 0:
                raise IndexError
        except IndexError:
            print("节点下标出界")
 
    def isEmpty(self):
        return self.vertices_num == 0
    def add_vertex(self, key):
        if key not in self.vertices:
            self.vertices.append(key)
            self.vertices_num+=1
        for i in range(len(self.mat)):
            self.mat[i].append(self.x)
  
        nRow = [self.x] * self.vertices_num
        self.mat.append(nRow)
    def add_edge(self,tail,head,val):#弧是从弧尾指向弧头的
        if head not in self.vertices:
            self.add_vertex(head)
        if tail not in self.vertices:
            self.add_vertex(tail)
        self.mat[self.vertices.index(tail)][self.vertices.index(head)]=val
        self.edges_list.append((tail,head,val))
        self.edges_dict[(tail,head)]=val
        self.edges_num+=1
    def pprint(self):
        for i in range(len(self.mat)):
            for j in range(len(self.mat)):
                if type(self.mat[i][j])==int:
                    print('{:^4d}'.format(self.mat[i][j]),end='')
                else:print('{:^4f}'.format(self.mat[i][j]),end='')
            print('\n')
            
        
    def chudu(self,node):
        count=0
        for val in self.mat[self.vertices.index(node)]:
            if (self.flag==1 and val<inf) or (self.flag==0 and val>0):
                count+=1
        return count
    def rudu(self,node):
        count=0
        for i in range(len(self.mat)):
            if (self.flag==1 and self.mat[i][self.vertices.index(node)]<inf) or (self.flag==0 and self.mat[i][self.vertices.index(node)]>0):
                count+=1
        return count
    def degree(self,node):
        return self.rudu(node)+self.chudu(node)

    # 无向图深度遍历
    def DFS1(self,node):
        visited=[False]*self.vertices_num
        res=[]
        def recur(res,node):
            i=self.vertices.index(node)
            if not visited[i]:
                visited[i]=True
                res.append(node)
            for j in range(len(self.mat)):
                if (self.flag==0 and self.mat[i][j]>0) or (self.flag==1 and self.mat[i][j]<inf) and not visited[j]:
                    recur(res,self.vertices[j])
        recur(res,node)
        return res
    # 有向图深度遍历
    def DFS2(self,node):
        i=self.vertices.index(node)
        
        def recur(node):
            i=self.vertices.index(node)
            if not visited[i]:
                visited[i]=True
                res.append(node)
            for j in range(len(self.mat)):
                if (self.flag==0 and self.mat[i][j]>0) or (self.flag==1 and self.mat[i][j]<inf) and not visited[j]:
                    recur(self.vertices[j])
        visited=[False]*len(self.vertices)
        res=[]
        recur(node)
        for val in (self.vertices[:i]+self.vertices[:i+1]):
            if not visited[self.vertices.index(val)]:
                recur(val)
        return res
    # 无向图广度遍历
    def BFS1(self,node):
        visited=[False]*len(self.vertices)
        res=[]
        queue=[]
        queue.append(node)
        visited[self.vertices.index(node)]=True
        while len(queue)!=0:
            tmp=queue.pop(0)
            res.append(tmp)
            i=self.vertices.index(tmp)
            for j in range(len(self.mat)):
                if (self.flag==0 and self.mat[i][j]>0) or (self.flag==1 and self.mat[i][j]<inf) and not visited[j]:
                    queue.append(self.vertices[j])
                    visited[j]=True
        return res
    
    # 有向图广度遍历
    def BFS2(self,node):
        visited=[False]*len(self.vertices)
        res=[]
        def tra(node):
            queue=[]
            queue.append(node)
            visited[self.vertices.index(node)]=True
            while len(queue)!=0:
                tmp=queue.pop(0)
                res.append(tmp)
                i=self.vertices.index(tmp)
                for j in range(len(self.mat)):
                    if (self.flag==0 and self.mat[i][j]>0) or (self.flag==1 and self.mat[i][j]<inf) and not visited[j]:
                        queue.append(self.vertices[j])
                        visited[j]=True
        tra(node)
        i=self.vertices.index(node)
        for val in (self.vertices[:i]+self.vertices[i+1:]):
            if not visited[self.vertices.index(val)]:
                tra(val)      
        return res





mat=[[inf,1,inf,3],[3,inf,6,inf],[inf,inf,inf,8],[inf,inf,4,inf]]
vertices=['A','B','C','D']
a=Graph(mat,vertices,1)
a.pprint()
# print(a.rudu('C'),' ',a.chudu('C'),' ',a.degree('C'))
# a.add_edge('A','C',2)
# a.pprint()
# print(a.rudu('C'),' ',a.chudu('C'),' ',a.degree('C'))
# a.add_edge('D','E',2)
# a.pprint()
# print(a.rudu('C'),' ',a.chudu('C'),' ',a.degree('C'))
# print(a.DFS1('D'))
# print(a.DFS2('D'))
# print(a.BFS1('D'))
# print(a.BFS2('D'))
print(a.DFS1('B'))
print(a.DFS2('B'))
print(a.BFS1('B'))
print(a.BFS2('B'))
