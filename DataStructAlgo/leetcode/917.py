class Solution:
    def reverseOnlyLetters(self, s: str) -> str:
        if not s:
            return "" 
        left,right,s = 0,len(s)-1,list(s)
        while left < right:
            while left < right and not(s[left].isalpha()): left+=1
            while left < right and not(s[right].isalpha()): right-=1
            s[left],s[right]=s[right],s[left]
            left+=1
            right-=1
        return ''.join(s)