class Solution:
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        candidates.sort()
        
        ret = []
        def helper(cand,t,idx):
            if t == 0:
                ret.append(cand)
            if t < 0:
                return 
            for i in range(idx,len(candidates)):
                if i > idx and candidates[i]==candidates[i-1]:
                    continue 
                helper(cand+[candidates[i]],t-candidates[i],i+1)
        helper([],target,0)
        return ret