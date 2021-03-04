import collections
class Solution:
    def missingNumber(self, nums: List[int]) -> int:
        ret =0 
        for i in range(len(nums)):
            ret ^= i
            ret ^= nums[i]
            
        ret ^=len(nums)
        return ret
        #if not nums: return
        #for i in range(len(nums)+1):
        #    if i not in nums:
        #        return i