class Solution:
    def merge(self, lst: List[List[int]]) -> List[List[int]]:
        if not lst: return []
        lst.sort(key = lambda x:x[0])
        ret = []
        last = lst[0]
        for cur in lst:
            if last[1]>=cur[0]:
                last[1] = max(cur[1],last[1])
            else:
                ret.append(last)
                last=cur
        ret.append(last)
        return ret