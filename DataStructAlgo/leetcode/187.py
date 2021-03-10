class Solution:
    def findRepeatedDnaSequences(self, s: str) -> List[str]:
        dict = {}
        ret = []
        for i in range(0,len(s)-9,1):
            if s[i:i+10] in dict:
                dict[s[i:i+10]]+=1
            else:
                dict[s[i:i+10]]=1
            if dict[s[i:i+10]]==2:
                ret.append(s[i:i+10])
        return ret