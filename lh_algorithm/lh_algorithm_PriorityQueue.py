def max_heapify(n, A, i):
    #最大堆化
    l, r = 2 * i + 1, 2 * i + 2
    largest = i
    if l < n and A[l] > A[i]:
        largest = l
    if r < n and A[r] > A[largest]:
        largest = r
    if i != largest:
        A[i], A[largest] = A[largest], A[i]
        max_heapify(n, A, largest)

def heap_maximum(A):
    #返回最大值即根节点
    return A[0] if A else - 1

def heap_extract_max(A, n):
    #最大值出队
    if n < 1:
        print('underflow')
    max = A[0]
    A[0] = A[n - 1]
    A.pop()
    n -= 1
    max_heapify(n, A, 0)
    return max

def heap_increase_key(i,A,key):
    #添加新元素并最大堆化
    if key < A[i]:
        print('new key is samller than current key')
    A[i] = key
    k = (i - 1) // 2  #key的根节点
    while i > 0 and A[k] < A[i]:
        A[k], A[i] = A[i], A[k]
        i = k#key上浮
        k = (i - 1) // 2#上浮后的根节点

def max_heap_insert(n, A, key):
    #添加新元素
    n += 1
    A.append(-float('inf'))
    heap_increase_key(n - 1, A, key)#n-1是当前最后一个元素的index
    

B = [3, 17, 27, 16, 13, 10, 1, 5, 7, 12, 4, 8, 9, 0]
A = []

for i in B:
    max_heap_insert(len(A), A, i)
    print(A)
n = len(A)
while n:
    print(heap_extract_max(A, n))
    n-=1