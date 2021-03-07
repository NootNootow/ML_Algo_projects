# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
   # def __init__(self):
    #    self.ret = []
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        if not root:
            return 
        stack = []
        ret=[]
        curr = root
        while stack or curr:
            if curr:
                stack.append(curr)
                curr=curr.left
            elif stack:
                curr=stack.pop()
                ret.append(curr.val)
                curr=curr.right
        return ret
        
    #    self.inorderTraversal(root.left)
     #   self.ret.append(root.val)
     #   self.inorderTraversal(root.right)
        
      #  return self.ret