import sys
class Solution:
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        if sum(nums)<target:
            return 0
        ret = sys.maxsize
        tmp,left,right=0,0,0
        while right < len(nums):
            tmp+=nums[right]
            while target<=tmp:
                ret = min(ret,right-left+1)
                tmp-=nums[left]
                left+=1
            right+=1
        return 0 if ret == sys.maxsize else ret
       # min_ = sys.maxsize
       # left , right = 0 ,1 
       # while left < len(nums):
        #    if sum((nums[left:right])) < s:
        #        if  right ==len(nums):
        #           break
        #        right +=1 
        #    else:
          #      if min_ > right - left:
            #        min_=right-left 
            #    left+=1
        #return min_ if min_!=sys.maxsize else 0