# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
import collections
class Solution:
    def invertTree(self, root: TreeNode) -> TreeNode:
        #if root != None:
            #temp = root.left
           # root.left = self.invertTree(root.right)
           #root.right = self.invertTree(temp)
        #    root.left,root.right = self.invertTree(root.right),self.invertTree(root.left)
        #queue = collections.deque([root])
        #while queue:
        #    node = queue.popleft()
        #    if node:
         #       node.left,node.right=node.right,node.left
         #       queue.append(node.left)
          #      queue.append(node.right)
        if not root :
            return 
        stack = [root]
        while stack:
            node= stack.pop()
            if node:
                node.left,node.right=node.right,node.left
                stack.extend([node.left,node.right])
        return root