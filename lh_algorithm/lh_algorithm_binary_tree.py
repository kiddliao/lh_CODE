class TreeNode:
    def __init__(self,x):
        self.val=x
        self.left=None
        self.right=None
#非递归的前序遍历，进栈顺序是前序遍历，出栈顺序中序遍历
#https://blog.csdn.net/monster_ii/article/details/82115772
class newTree:
    def __init__(self,x):
        self.head=self.new(None,x,0)
        self.pre,self.mid,self.post=[],[],[]
        self.pre2,self.mid2,self.post2=[],[],[]
        self.ceng=[]
    def new(self,tree,x,i):
        if i<len(x):
            if x[i]==None:return TreeNode(None)
            else:
                tree=TreeNode(x[i])
                tree.left=self.new(tree.left,x,2*i+1)
                tree.right=self.new(tree.right,x,2*i+2)
                return tree
            return tree
    def preOrderTraverse(self,node):
        if node==None:
            return
        self.pre.append(node.val)
        self.preOrderTraverse(node.left)
        self.preOrderTraverse(node.right)
    def preOrderTraverse2(self,node):
        stack=[]
        cur=node
        while(cur!=None or len(stack)!=0):
            while(cur!=None):
                stack.append(cur)
                self.pre2.append(cur.val)
                cur=cur.left
            top=stack[-1]
            stack.pop()
            cur=top.right

    def midOrderTraverse(self,node):
        if node==None:
            return
        self.midOrderTraverse(node.left)
        self.mid.append(node.val)
        self.midOrderTraverse(node.right)
    def midOrderTraverse2(self,node):
        stack=[]
        cur=node
        while(cur!=None or len(stack)!=0):
            while(cur!=None):
                stack.append(cur)
                cur=cur.left
            top=stack[-1]
            self.mid2.append(top.val)
            stack.pop()
            cur=top.right

    def postOrderTraverse(self,node):
        if node==None:
            return
        self.postOrderTraverse(node.left)
        self.postOrderTraverse(node.right)
        self.post.append(node.val)
    
    def postOrderTraverse2(self,node):
        stack=[]
        cur=node
        last=None
        while(cur!=None or len(stack)!=0):
            while(cur!=None):
                stack.append(cur)
                cur=cur.left
            top=stack[-1]
            if top.right==None or top.right==last:
                self.post2.append(top.val)
                stack.pop()
                last=top
            else:
                cur=top.right
    def cengOrderTraverse(self,node):
        queue=[]
        cur=node
        if cur:
            queue.append(cur)
        while(queue):
            top=queue[0]
            self.ceng.append(top.val)
            queue.pop(0)
            if top.left:
                queue.append(top.left)
            if top.right:
                queue.append(top.right)



# a=newTree([1,2,3,4,5])
# a.preOrderTraverse(a.head)
# a.midOrderTraverse(a.head)
# a.postOrderTraverse(a.head)
# print(a.pre)
# print(a.mid)
# print(a.post)
# a.preOrderTraverse2(a.head)
# a.midOrderTraverse2(a.head)
# a.postOrderTraverse2(a.head)
# print(a.pre2)
# print(a.mid2)
# print(a.post2)
# a.cengOrderTraverse(a.head)
# print(a.ceng)


