# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def addOneRow(self, root: TreeNode, v: int, d: int) -> TreeNode:
        if not root:
            return
        if d == 1:
            tmp = TreeNode(v)
            tmp.left = root
            return tmp 
        queue = [root]
        while root and d>2:
            n = len(queue)
            for i in range(n):
                curr= queue.pop(0)
                if curr.left:
                    queue.append(curr.left)
                if curr.right:
                    queue.append(curr.right)
            d-=1
        for i in range(len(queue)):
            curr = queue.pop(0)
            curr.left,curr.right,curr.left.left,curr.right.right = TreeNode(v),TreeNode(v),curr.left,curr.right
        return root
            