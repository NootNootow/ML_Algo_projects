class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        def dfs(grd,i,j):
            if i < 0 or j <0 or i>=len(grd) or j >=len(grd[0]) or \
                grd[i][j]!="1":
                return
            grd[i][j]='#'
            dfs(grd,i,j+1)
            dfs(grd,i,j-1)
            dfs(grd,i+1,j)
            dfs(grd,i-1,j)
        count = 0 
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j]=="1":
                    dfs(grid,i,j)
                    count+=1
        return count