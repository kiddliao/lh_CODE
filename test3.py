def core(matrix,m,n):
    start = []
    end = []
    for i in range(m):
        for j in range(n):
            if matrix[i][j] == 'S':
                start = [i, j]
            if matrix[i][j] == 'E':
                end = [i, j]
    if not start or not end:
        return False
    queue = [start]
    visited = set()
    visited.add(tuple(start))
    while queue:
        if end in queue:
            return True
        x, y = queue.pop(0)
        for ix, iy in [[x + 1, y], [x - 1, y], [x, y + 1], [x, y - 1]]:
            if 0 <= ix < m and 0 <= iy < n and (ix,iy) not in visited and matrix[ix][iy] in '.E':
                queue.append([ix, iy])
                visited.add((ix,iy))
    return False
    

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        m, n = list(map(int, input().split()))
        matrix = []
        for i in range(m):
            matrix.append(list(input().strip()))
        res = core(matrix,m,n)
        if res:
            print('YES')
        else:
            print('NO')