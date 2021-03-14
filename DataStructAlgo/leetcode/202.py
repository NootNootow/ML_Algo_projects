class Solution:
    def isHappy(self, n: int) -> bool:
        dict = {}
        while n!=1 :
            tmp=0
            while n:
                tmp+= (n%10)**2
                n//=10
            if tmp in dict:
                return False
            else:
                dict[tmp]=1
            n=tmp 
        return True