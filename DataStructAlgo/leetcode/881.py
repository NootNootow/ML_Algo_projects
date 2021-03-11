class Solution:
    def numRescueBoats(self, ppl: List[int], lim: int) -> int:
        ppl.sort()
        ret=0         
        l,r=0,len(ppl)-1 
        while l <=r:
            if ppl[l]+ppl[r]<=lim:
                l+=1
                r-=1
            else:
                r-=1
            ret+=1
        return ret