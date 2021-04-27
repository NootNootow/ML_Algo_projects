class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        if not nums: return 0
        ret = 0 
        tmp = set(nums)
        for n in nums:
            if n-1 not in tmp:
                curr = 1 
                while n+1 in tmp:
                    curr +=1
                    n=n+1
                ret = max(ret,curr)
        return ret