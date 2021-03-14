# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def swapNodes(self, head: ListNode, k: int) -> ListNode:
        if not head: return 
        first,cur =None,head
        for _ in range(k-1):
            cur= cur.next
        first = cur
        last = head 
        while cur.next:
            cur=cur.next
            last=last.next
        first.val,last.val=last.val,first.val
        return head