import math
if __name__ == '__main__':
    n = int(input())
    nums = list(map(int, input().split()))
    if n == 1:
        if nums[0] == 0:
            print('No')
        else:
            print('{:.2f}'.format(-nums[1]/nums[0]))
    elif n == 2:
        a, b, c = nums
        tmp = b ** 2 - 4 * a * c
        if tmp < 0:
            print('No')
        elif tmp == 0:
            x = (-b + tmp) / (2 * a)
            print('{:.2f}'.format(x))
        else:
            x1 = (-b + tmp) / (2 * a)
            x2 = (-b - tmp) / (2 * a)
            print('{:.2f} {:.2f}'.format(x1, x2))
    if n == 3:
        a, b, c, d = nums
        A = b ** 2 - 3 * a * c
        B = b * c - 9 * a * d
        C = c ** 2 - 3 * b * d
        D = B ** 2 - 4 * A * C
        if A == B == 0:
            print('{.2f}'.format(-c / b))
        elif D > 0:
            Y1 = A*b + 3 * a * (1 / 2 * (-B + (D ** (1 / 2))))
            Y2 = A*b + 3 * a * (1 / 2 * (-B - (D ** (1 / 2))))
            print('{:.2f}'.format(1 / (3 * a) * (-b - Y1 ** (1 / 3) - Y2 ** (1 / 3))))
        elif D == 0:
            K = B / A
            print('{:.2f} {:.2f}'.format(-b / a + K, -K / 2))
