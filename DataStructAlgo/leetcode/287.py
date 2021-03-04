class Solution:
    def findDuplicate(self, nums: List[int]) -> int:
        visited = [0]*len(nums)
        for i in range(len(nums)):
            if visited[nums[i]]==1:
                return nums[i]
            else:
                visited[nums[i]]=1
        print(visited)
        
        #slow=fast = nums[0]
        #while True:
        #    slow = nums[slow]
        #    fast = nums[nums[fast]]
        #    if slow == fast: break
        #slow = nums[0]
        #while slow!=fast:
         #   slow=nums[slow]
          #  fast=nums[fast]
        #return fast