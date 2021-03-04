# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def helper(self,nums,left,right):
        if left>right: 
            return None
        mid=left+(right-left)//2
        Node = TreeNode(nums[mid])
        Node.left=self.helper(nums,left,mid-1)
        Node.right=self.helper(nums,mid+1,right)
        return Node
    def sortedArrayToBST(self, nums: List[int]) -> TreeNode:
        if not nums: return 
        return self.helper(nums,0,len(nums)-1)
        
        #if not nums:
       #     return 
        #mid = len(nums)//2
        #root = TreeNode(nums[mid])
        #root.left = self.sortedArrayToBST(nums[:mid])
        #root.right = self.sortedArrayToBST(nums[mid+1:])
       # return root