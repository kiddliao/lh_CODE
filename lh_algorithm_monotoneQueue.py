nums = [1, 3, -1, -3, 5, 3, 6, 7]
window = 3
def monotoneQueue(nums, window):
    if len(nums) <= window:
        return [max(nums)], [min(nums)]
    imax, imin = [], []
    nums = ['*'] + nums
    queue1, queue2 = [0], [0]
    for i in range(1, len(nums)):
        if len(queue1) > 1 and i - queue1[1] >= window:
            queue1.pop(1)
        
        if len(queue2) > 1 and i - queue2[1] >= window:
            queue2.pop(1)

        while nums[queue1[-1]] != '*' and nums[i] >= nums[queue1[-1]]:
            queue1.pop()
        queue1.append(i)

        while nums[queue2[-1]] != '*' and nums[i] <= nums[queue2[-1]]:
            queue2.pop()
        queue2.append(i)

        if i >= window:
            imax.append(nums[queue1[1]])
            imin.append(nums[queue2[1]])
    
    return imax, imin
    
print(monotoneQueue(nums, window))



        
        
