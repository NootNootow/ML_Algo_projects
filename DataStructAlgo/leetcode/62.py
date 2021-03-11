class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        lst =[[0 for _ in range(n)] for _ in range(m)]
        for i in range(m):
            lst[i][0]=1
        for i in range(n):
            lst[0][i]=1
        for i in range(1,m):
            for j in range(1,n):
                lst[i][j]=lst[i-1][j]+lst[i][j-1]
        return lst[m-1][n-1]