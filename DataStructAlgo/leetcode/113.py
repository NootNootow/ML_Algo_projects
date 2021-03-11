# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def pathSum(self, root: TreeNode, targetSum: int) -> List[List[int]]:
        self.ret = [] 
        def helper(root,target,current):
            if not root: return 
            if target == root.val and not root.right and not root.left:
                self.ret.append(current+[root.val])
            helper(root.left, target-root.val, current + [root.val])
            helper(root.right, target-root.val, current + [root.val])
        helper(root,targetSum,[])
        return self.ret