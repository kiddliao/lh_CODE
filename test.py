if __name__=='__main__':
    m, n = 4,3
    nums = list(range(1, m * n + 1))
    matrix = [[0] * n for _ in range(m)]
    count = 0
    for x in range(m):
        i,j=x,0
        while 0 <= j < n and 0 <= i < m:
            matrix[i][j] = nums[count]
            count += 1
            i -= 1
            j += 1
    for x in range(1, n):
        i, j = m - 1, x
        while 0 <= j < n and 0 <= i < m:
            matrix[i][j] = nums[count]
            count += 1
            i -= 1
            j += 1



    

