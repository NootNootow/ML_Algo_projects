class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        #for i,num in enumerate(sorted(set(nums))):
        #    nums[i]=num
        #return len(set(nums))
        temp = 0
        for i in range(1,len(nums)):
            if nums[temp]!=nums[i]:
                temp+=1
                nums[temp]=nums[i]
                
        return temp+1