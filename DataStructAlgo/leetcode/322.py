class Solution:
    def coinChange(self, coins: List[int], amt: int) -> int:
        dp = [amt+1]*(amt+1)
        dp[0]=0
        for i in range(amt+1):
            for j in range(len(coins)):
                if coins[j]<=i:
                    dp[i]=min(dp[i],1+dp[i-coins[j]])
        return -1 if dp[-1]==amt+1 else dp[-1]