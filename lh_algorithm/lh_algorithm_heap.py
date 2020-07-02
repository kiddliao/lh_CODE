def max_heapify(n, A, i):
    #获取左右儿子的index
    l, r = 2 * i + 1, 2 * i + 2
    #存储根和左右子节点的最大值的index
    largest = i
    if l < n and A[l] > A[i]:
        largest = l
    if r < n and A[r] > A[largest]:
        largest = r
    #如果根节点的值不是最大的就交换 使得根节点的值最大 largest就是之前根节点的index
    #以largest为根节点继续遍历堆化 假设l左子树的值大于原根节点的值 交换后 因为不能保证原根节点要比l的子节点大所以继续遍历
    if i != largest:
        A[i], A[largest] = A[largest], A[i]
        max_heapify(n,A,largest)



def build_max_heap(n, A):
    #从最后一个内部节点开始自底向上遍历
    for i in range((n - 1) // 2, -1, -1):
        max_heapify(n,A,i)


def heapsort(A):
    #首先建堆
    n = len(A)
    build_max_heap(n, A)
    #建堆完成后把最大值依次移除并重新建堆
    curlen = n
    for i in range(n-1, 0, -1):
        A[0], A[i] = A[i], A[0]
        curlen-=1
        max_heapify(curlen,A,0)



A = [27,17,3,16,13,10,1,5,7,12,4,8,9,0]
heapsort(A)
print (A)