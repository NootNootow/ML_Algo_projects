class Solution:
    def longestIncreasingPath(self, matrix: List[List[int]]) -> int:
        lst = [(0,1),(0,-1),(1,0),(-1,0)]
        rows,col = len(matrix),len(matrix[0])
        cache = [[None]*col for _ in range(rows)]
        longest = 0 
        def dfs(x,y):
            if cache[x][y]: return cache[x][y]
            longest = 0
            for i,j in lst:
                newX,newY = x+i,y+j
                if newX>=0 and newX<rows and newY>=0 and newY<col \
                    and matrix[x][y]<matrix[newX][newY]:
                    longest = max(longest,dfs(newX,newY))
            cache[x][y] = longest+1
            return cache[x][y]
        for i in range(rows):
            for j in range(col):
                longest = max(longest,dfs(i,j))
        return longest