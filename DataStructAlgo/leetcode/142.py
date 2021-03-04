# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def detectCycle(self, head: ListNode) -> ListNode:
        if not head:
            return None
        if not head.next:
            return None
        slow,fast = head,head
        while fast and fast.next:
            slow=slow.next
            fast=fast.next.next
            if slow == fast:
                break
        if slow == fast:
            while head!=fast:
                head=head.next
                fast=fast.next
            return head
        return None