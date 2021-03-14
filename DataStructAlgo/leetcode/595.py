"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""

class Solution:
    """
    @param root: the root of binary tree
    @return: the length of the longest consecutive sequence path
    """
    def longestConsecutive(self, root):
        # write your code her
        
        if not root: return
        self.ret = 0
        def helper(root,curr,search):
            if not root:
                return
            if root.val == search:
                curr+=1
            else:
                curr=1
            self.ret = max(self.ret,curr)
            helper(root.left,curr,root.val+1)
            helper(root.right,curr,root.val+1)

        helper(root,0,0)
        return self.ret