class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        if not board:
            return False
        n,m=len(board),len(board[0])
        def dfs(i,j,board,word,idx):
            if idx==len(word):
                return True
            if i<0 or i>=n or j<0 or j>=m or word[idx]!=board[i][j]:
                return False
            tmp = board[i][j]
            board[i][j]='#'
            ret = dfs(i+1,j,board,word,idx+1) or dfs(i,j+1,board,word,idx+1) or \
                dfs(i-1,j,board,word,idx+1) or dfs(i,j-1,board,word,idx+1)
            board [i][j]=tmp
            return ret
        
        for i in range(n):
            for j in range(m):
                if dfs(i,j,board,word,0):
                    return True
                
        return False