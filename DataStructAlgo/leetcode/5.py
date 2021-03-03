class Solution:
    def helper(self,left,right,s):
        if left is None or left>right: return 0
        while left>=0 and right<len(s) and s[left]==s[right]:
            left-=1
            right+=1
        return right-left-1
    def longestPalindrome(self, s: str) -> str:
        if not s:
            return
        tmp = 0 
        left,right=0,0
        for i in range(len(s)):
            odd = self.helper(i,i,s)
            even= self.helper(i,i+1,s)
            tmp = max(odd,even)
            if tmp > right-left:
                left= i-((tmp-1)//2)
                right=i+(tmp//2)
        return s[left:right+1]