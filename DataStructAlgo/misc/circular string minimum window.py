import collections 
def min_window(s:str,target:str):
	min_win = len(s) +1
	lst = [s[i:]+s[:i] for i in range(len(s))]
	for s in lst:
		left,right = 0,0
		d = collections.Counter(target)
		len_s,len_t = len(s),len(target)
		while right < len_s:
			if s[right] in d:
				if d[s[right]]>0:
					len_t-=1
				d[s[right]]-=1
			while len_t ==0:
				if right - left +1 < min_win:
					min_win = right-left+1
				if s[left] in d:
					d[s[left]]+=1
					if d[s[left]] > 0 :
						len_t+=1
				left +=1
			right+=1
	return min_win