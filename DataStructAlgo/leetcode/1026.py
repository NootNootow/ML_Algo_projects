# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maxAncestorDiff(self, root: TreeNode) -> int:
        if not root:
            return 
        low,high =float('inf'),-float('inf')
        self.tmp = 0
        
        def helper(root,low,high):
            if not root:
                self.tmp= max(self.tmp,abs(high-low))
                return 
            helper(root.left,min(root.val,low),max(root.val,high))
            helper(root.right,min(root.val,low),max(root.val,high))
        helper(root,low,high)
        return self.tmp