# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isSubtree(self, s: TreeNode, t: TreeNode) -> bool:
        def helper(root,tree):
            if not root or not tree:
                return root is None and tree is None
            elif root.val==tree.val:
                return helper(root.left,tree.left) and helper(root.right,tree.right)
            else:
                False
        if not s:
            return False
        elif helper(s,t):
            return True
        else:
            return self.isSubtree(s.left,t) or self.isSubtree(s.right,t)
        