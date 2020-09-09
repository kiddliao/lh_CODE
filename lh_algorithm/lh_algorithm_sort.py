from math import log10
import numpy as np
import time
#1插入排序
#1.1直接插入排序
#稳定排序 平均时间复杂度O(n^2) 当完全有序时有最好时间复杂度为O(n) 最坏时间复杂度为O(n^2) 空间复杂度为O(1)
#适用于初始列表基本有序的情况,当初始列表无序且n较大不宜使用
#适用于顺序结构和链式结构
def InsertSort(nums):
    n = len(nums)
    for i in range(1, n):
        if nums[i-1] > nums[i]:
            #寻找插入位置
            tmp = nums[i]
            j = i - 1
            while j >= 0 and nums[j] > tmp:
                #后挪
                nums[j + 1] = nums[j]
                j -= 1
            nums[j + 1] = tmp
    return nums


#1.2折半插入排序
#稳定排序 平均时间复杂度O(n^2) 当完全有序时有最好时间复杂度为O(n) 最坏时间复杂度(虽然比较复杂度是O(logn)但是移动元素复杂度还是O(n))为O(n^2) 空间复杂度为O(1)
#平均性能强于直接插入排序,适用于初始列表无序且n较大的情况(比较次数少)
#仅适用于顺序结构(二分查找需要任意访问任何元素)
def BinaryInsertSort(nums):
    n = len(nums)
    for i in range(1, n):
        if nums[i-1] > nums[i]:
            #寻找插入位置
            tmp = nums[i]
            l, r = 0, i - 1
            #寻找最后一个不大于tmp的元素的位置
            while l + 1 < r:
                m = (r - l) // 2 + l
                if nums[m] > tmp:
                    r = m
                elif nums[m] <= tmp:
                    l = m
            ind = -1
            if nums[r] <= tmp:
                ind = r
            elif nums[l] <= tmp:
                ind = l
            #挪元素
            for k in range(i - 1, ind, -1):
                nums[k + 1] = nums[k]
            nums[k] = tmp
    return nums


#1.3希尔排序
#不稳定排序 平均时间复杂度O(n^1.3) 当完全有序时有最好时间复杂度为O(n) 最坏时间复杂度为O(n^2) 空间复杂度为O(1)
#间隔相同的元素为一组,同组进行直接插入排序,也就是让列表符合直接插入排序最适用的情况(基本有序)
#增量序列的最佳选择涉及到数学上至今未解决的难题,到目前尚未有人求得一种最好的增量序列,时间复杂度也只是近似
#增量序列一般使得列表中的值没有除1以外的其他公因子,且最后的增量必须是1
#适用于初始列表无序且n较大的情况(比较次数少)
#仅适用于顺序结构(跳跃式移动需要任意访问任何元素)
def ShellInsert(nums, k):
    n = len(nums)
    for i in range(k):
        cur = []
        j = i
        while j < n:
            cur.append(nums[j])
            j += k
        cur = InsertSort(cur)
        j = i
        count = 0
        while j < n:
            nums[j] = cur[count]
            j += k
            count += 1   

def ShellSort(nums, dt=[5, 3, 1]):
    for k in dt:
        ShellInsert(nums, k)
    return nums


#2交换排序
#2.1冒泡排序
#稳定排序 平均时间复杂度O(n^2) 当完全有序时有最好时间复杂度为O(n) 最坏时间复杂度为O(n^2) 空间复杂度为O(1)
#适用于顺序结构和链式结构
#平均性能差于直接插入排序,当初始列表无序且n较大不宜使用
def BubbleSort(nums):
    n = len(nums) 
    for i in range(n):
        #flag表示当前迭代有没有进行交换,如果没有就是有序,不进行后序迭代
        flag = 0
        for j in range(n - i - 1):
            if nums[j] > nums[j+1]:
                flag = 1
                nums[j], nums[j+1] = nums[j+1], nums[j]
        if flag == 0:
            break
        else:
            flag = 0
    return nums


#2.2快速排序
#不稳定排序 平均时间复杂度O(nlogn) 当每趟排序都可以把列表均匀地分成两个长度大致相等的子列表时有最好时间复杂度为O(nlogn)
#当排序树是单支树也就是列表正序或逆序最坏时间复杂度为O(n^2) 空间复杂度为O(nlogn)
#仅适用于顺序结构(需要任意访问任何元素)
#当n较大时,快速排序是所有内部排序方法中速度最快的,适用于初始列表无序且n较大的情况
def QuickParaition(nums, l, r):
    low, high = l, r
    tmp = nums[low]
    while low < high:
        while low<high and nums[high] >= tmp:
            high -= 1
        nums[low] = nums[high]
        while low < high and nums[low] <= tmp:
            low += 1
        nums[high] = nums[low]
    assert low == high
    nums[low] = tmp
    return low
            
def QSort(nums, l, r):
    if l < r:
        ind = QuickParaition(nums, l, r)
        QSort(nums,l,ind)
        QSort(nums,ind+1,r)

def QuickSort(nums):
    n = len(nums)
    l, r = 0, n - 1
    QSort(nums, l, r)
    return nums


#3选择排序
#3.1简单选择排序
#稳定排序 平均时间复杂度O(n^2) 当完全有序时有最好时间复杂度为O(n) 最坏时间复杂度为O(n^2) 空间复杂度为O(1)
#适用于顺序结构和链式结构
#移动元素次数少,当每一记录占用的空间多的时候优于直接插入排序
def SelectSort(nums):
    n = len(nums)
    for i in range(n):
        curMin = float('inf')
        curInd = -1
        for j in range(i,n):
            if nums[j] < curMin:
                curMin = nums[j]
                curInd = j
        nums[i], nums[curInd] = nums[curInd], nums[i]
    return nums


#3.2树形选择排序
#不稳定排序 平均时间复杂度为O(nlogn) 但是耗费储存空间较多 且和当前最小元素比较次数过多 因此提出堆排序
#区别于简单选择排序O(n)遍历得到当前最小元素,树形选择排序两两比较元素得到当前最小元素,比较复杂度为O(logn)
#仅适用顺序结构

#3.3堆排序
#不稳定排序 平均时间复杂度为O(nlogn) 最好时间复杂度为O(nlogn) 最坏时间复杂度为O(nlogn) 空间复杂度为O(1)
#仅适用顺序结构
#因此初始建堆所需比较次数较多,因此n较小时不宜使用,因为最坏情况下时间复杂度还是O(nlogn),相比于快速排序最坏情况下时间复杂度O(n^2)是个优点,当n较大时堆排序比较高效
def Max_Heapify(n, nums, i):
    #获取左右儿子的index
    l, r = 2 * i + 1, 2 * i + 2
    #存储根和左右子节点的最大值的index
    largest = i
    if l < n and nums[l] > nums[i]:
        largest = l
    if r < n and nums[r] > nums[largest]:
        largest = r
    #如果根节点的值不是最大的就交换 使得根节点的值最大 largest就是之前根节点的index
    #以largest为根节点继续遍历堆化 假设l左子树的值大于原根节点的值 交换后 因为不能保证原根节点要比l的子节点大所以继续遍历
    if i != largest:
        nums[i], nums[largest] = nums[largest], nums[i]
        Max_Heapify(n, nums, largest)

def Build_Max_Heap(n, nums):
    #从最后一个内部节点开始自底向上遍历
    for i in range((n - 1) // 2, -1, -1):
        Max_Heapify(n, nums, i)

def HeapSort(nums):
    #初始建堆
    n = len(nums)
    Build_Max_Heap(n, nums)
    #建堆完成后把最大值依次移除并重新建堆
    curlen = n
    for i in range(n-1, 0, -1):
        nums[0], nums[i] = nums[i], nums[0]
        curlen-=1
        Max_Heapify(curlen, nums, 0)
    return nums


#4归并排序
#4.1二路归并排序
#稳定排序 平均时间复杂度O(nlogn) 当完全有序时有最好时间复杂度为O(nlogn) 最坏时间复杂度为O(nlogn) 使用了辅助数组,因此空间复杂度为O(n)
#适用于顺序结构和链式结构
def MSort(nums, l, r):
    if l >= r:
        return
    m = (r - l) // 2 + l
    MSort(nums, l, m)
    MSort(nums, m + 1, r)
    i, j = l, m + 1
    tmp = []
    while i <= m and j <= r:
        if nums[i] <= nums[j]:
            tmp.append(nums[i])
            i += 1
        else:
            tmp.append(nums[j])
            j += 1
    if i <= m:
        tmp += nums[i:m + 1]
    if j <= r:
        tmp += nums[j:r + 1]
    nums[l:r+1]=tmp[:]

def MergeSort(nums):
    n = len(nums)
    l, r = 0, n - 1
    MSort(nums, l, r)
    return nums    


#5基数排序
#假设每个数有d位,每位的取值可能有r个,每趟分配的时间复杂度为O(n),每趟收集的时间复杂度为O(rd)
##稳定排序 平均时间复杂度O(d(n+r)) 当完全有序时有最好时间复杂度为O(d(n+r)) 最坏时间复杂度为O(d(n+r)) 空间复杂度为O(r+n)
#适用于顺序结构和链式结构
class LinkNode:
    def __init__(self, val):
        self.val = val
        self.next = None
        
def RadixSort(nums):
    n=len(nums)
    d = int(log10(max(nums))) + 1
    r = 10
    for i in range(d - 1, -1, -1):
        head = [LinkNode(None) for i in range(r)]
        tail = head.copy()
        #分配
        for j in range(n):
            insert_ind = int(str(nums[j]).rjust(d, '0')[i])
            p = tail[insert_ind]
            new_node = LinkNode(nums[j])
            new_node.next = p.next
            p.next = new_node
            tail[insert_ind] = p.next
        #收集
        cur = []
        for j in range(r):
            p = head[j].next
            while p:
                cur.append(p.val)
                p = p.next
        nums[:] = cur[:]
    return nums

def test_time(f, nums):
    start = time.time()
    res = f(nums)
    end = time.time()
    print('{}耗时{}s,检测结果如下:\n{}'.format(repr(f).split()[1], end - start, res[:100]))
    

if __name__=='__main__':
    # nums = [27, 17, 3, 16, 13, 10, 1, 5, 7, 12, 4, 8, 9, 8, 0]
    # # nums = [49, 38, 65, 97, 76, 13, 27, 49, 55, 4]
    # # nums = [278, 109, 63, 930, 589, 184, 505, 269, 8, 83]
    # res = InsertSort(nums.copy())
    # print(res)
    # res = BinaryInsertSort(nums.copy())
    # print(res)
    # res = ShellSort(nums.copy())
    # print(res)
    # res = BubbleSort(nums.copy())
    # print(res)
    # res = QuickSort(nums.copy())
    # print(res)
    # res = SelectSort(nums.copy())
    # print(res)
    # res = HeapSort(nums.copy())
    # print(res)
    # res = MergeSort(nums.copy())
    # print(res)
    # res = RadixSort(nums.copy())
    # print(res)
    nums = np.arange(100000)
    np.random.shuffle(nums)
    nums = nums.tolist()
    function_list = [InsertSort, BinaryInsertSort, ShellSort, BubbleSort, QuickSort, SelectSort, HeapSort, MergeSort, RadixSort]
    for f in function_list:
        test_time(f, nums.copy())
    
