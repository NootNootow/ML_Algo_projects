class Solution:
    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        self.row, self.col = len(obstacleGrid)-1,len(obstacleGrid[0])-1
        self.dict = {}
        def dfs(grid,i,j):
            if str(i)+str(j) in self.dict:
                return self.dict[str(i)+str(j)]
            if i > self.row or j > self.col:
                return 0
            if grid[i][j]==1:
                return 0
            if i == self.row and j == self.col:
                return 1
            count = 0 
            count += dfs(grid,i+1,j)
            count += dfs(grid,i,j+1)
            self.dict[str(i)+str(j)]=count
            return count
        return dfs(obstacleGrid,0,0)