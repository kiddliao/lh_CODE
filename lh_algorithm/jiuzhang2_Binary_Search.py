#二分法模板
#通用模板不会死循环
def bs(nums.target):
    '''
    找到target的位置,nums不重复
    '''
    if not nums:
        return - 1
    start, end = 0, len(nums) - 1
    while start + 1 < end:
        mid = start + (end - start) // 2 
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            start = mid
        else:
            end = mid
    if nums[start] == target:
        return start
    if nums[end] == target:
        return end
    return - 1


def bs1(nums, target):
    '''
    find the first position of target
    '''
    if not nums:
        return - 1
    start, end = 0, len(nums) - 1
    while start + 1 < end:
        mid = start + (end - start) // 2
        if nums[mid] == target:
            end = mid
        elif nums[mid] < target:
            start = mid
        else:
            end = mid
    if nums[start] == target:
        return start
    if nums[end] == target:
        return end
    return - 1

def bs2(nums, target):
    '''
    find the last position of target
    '''
    if not nums:
        return - 1
    start, end = 0, len(nums) - 1
    while start + 1 < end:
        mid = start + (end - start) // 2
        if nums[mid] == target:
            start = mid
        elif nums[mid] < target:
            start = mid
        else:
            end = mid
    if nums[end] == target:
        return end
    if nums[start] == target:
        return start
    return - 1

def bs3(nums, target):
    '''
    找到最后一个不大于target的数的位置
    '''
    if not nums:
        return - 1
    start, end = 0, len(nums) - 1
    while start + 1 < end:
        mid = start + (end - start) // 2
        if nums[mid] == target:
            end = mid
        elif nums[mid] < target:
            start = mid
        else:
            end = mid 
    if nums[end] <= target:
        return end
    if nums[start] <= target:
        return start
    return - 1

def bs4(nums, target):
    '''
    找到第一个不小于target的数的位置
    '''
    if not nums:
        return - 1
    start, end = 0, len(nums) - 1
    while start + 1 < end:
        mid = start + (end - start) // 2
        if nums[mid] == target:
            end = mid
        elif nums[mid] < target:
            start = mid
        else:
            end = mid 
    if nums[start] >= target:
        return start
    if nums[end] >= target:
        return end
    return - 1

nums = [1, 1, 1, 2, 3, 3, 3]
# print(bs1(nums, 1))
# print(bs2(nums, 1))
# print(bs1(nums, 2))
# print(bs2(nums, 2))
# print(bs1(nums, 3))
# print(bs2(nums, 3))
print(bs3(nums, 0.5))
print(bs4(nums, 0.5))
print(bs3(nums, 1.5))
print(bs4(nums, 1.5))
print(bs3(nums, 2.5))
print(bs4(nums, 2.5))
print(bs3(nums, 3.5))
print(bs4(nums, 3.5))
         