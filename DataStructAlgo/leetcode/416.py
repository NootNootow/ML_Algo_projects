class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        if not nums: return False 
        total = sum(nums)
        if total%2!=0:
            return False
        
        def helper(nums,idx,curr_sum,total,dict):
            if str(idx)+str(curr_sum) in dict:
                return dict[str(idx)+str(curr_sum)]
            if curr_sum *2==total:
                return True
            if curr_sum*2>total or idx>=len(nums):
                return False 
            found = helper(nums,idx+1,curr_sum,total,dict) or \
                    helper(nums,idx+1,curr_sum+nums[idx],total,dict)
                
            dict[str(idx)+str(curr_sum)] = found 
            return found
        
        return helper(nums,0,0,total,{})