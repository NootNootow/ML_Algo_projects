class Solution:
    def reverseVowels(self, s: str) -> str:
        if not s: return ""
        s = list(s)
        vowels = set(['a','A','e','E','i','I','o','O','u','U'])
        i,j =0,len(s)-1
        while i < j:
            while i < j and s[i] not in vowels: i+=1
            while j > i and s[j] not in vowels: j-=1
            s[i],s[j]=s[j],s[i]
            i+=1
            j-=1
        return ''.join(s)