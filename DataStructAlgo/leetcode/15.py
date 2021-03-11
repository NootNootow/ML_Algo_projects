class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        if not nums or len(nums)<3:
            return []
        nums.sort()
        ret,size = [],len(nums)
        for i,n in enumerate(nums[:size-2]):
            if i > 0 and nums[i]==nums[i-1]: continue
            left,right,sum = i+1,size-1,0-n
            while left < right:
                if nums[left]+nums[right]==sum:
                    ret.append([n,nums[left],nums[right]])
                    while left < right and nums[left]==nums[left+1]: left+=1
                    while left < right and nums[right]==nums[right-1]: right-=1
                    left+=1
                    right-=1
                elif nums[left]+nums[right]>sum:
                    right-=1
                else:
                    left+=1
        return ret