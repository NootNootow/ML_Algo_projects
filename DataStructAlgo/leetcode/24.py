# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def swapPairs(self, head: ListNode) -> ListNode:
        if not head or not head.next:
            return head
        a,b=head,head.next
        while b:
            a.val,b.val=b.val,a.val
            if not b.next:
                return head
            a=a.next.next
            b=b.next.next
        return head