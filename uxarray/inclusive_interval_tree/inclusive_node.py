from intervaltree.node import Node
from .inclusive_interval import InclusiveInterval

class InclusiveNode(Node):
    def __init__(self,
                 x_center=None,
                 s_center=set(),
                 left_node=None,
                 right_node=None):
        super(InclusiveNode, self).__init__(x_center, s_center, left_node, right_node)

    @classmethod
    def from_interval(cls, interval):
        """
        :rtype : change to InclusiveNode from intervaltree.Node
        """
        center = interval.begin
        return InclusiveNode(center, [interval])

    @classmethod
    def from_intervals(cls, intervals):
        """
        :rtype : change to InclusiveNode from intervaltree.Node
        """
        if not intervals:
            return None
        return InclusiveNode.from_sorted_intervals(sorted(intervals))

    @classmethod
    def from_sorted_intervals(cls, intervals):
        """
        :rtype : change to InclusiveNode from intervaltree.Node
        """
        if not intervals:
            return None
        node = InclusiveNode
        node = node.init_from_sorted(intervals)
        return node

    def center_hit(self, inclusive_interval: InclusiveInterval):
        """Returns whether interval overlaps self.x_center, inclusive."""
        return inclusive_interval.contains_point(self.x_center)

    def search_point(self, point, result):
        """
        Tweak the parent class implementation that it returns all intervals that contain point. Begin and end are inclusive.
        """
        for k in self.s_center:
            if k.begin <= point <= k.end:
                result.add(k)
        if point < self.x_center and self[0]:
            return self[0].search_point(point, result)
        elif point > self.x_center and self[1]:
            return self[1].search_point(point, result)
        return result

    def add(self, interval):
        """
        Tweak the parent class implementation by calling the InclusiveNode.from_interval() method.
        Returns self after adding the interval and balancing.
        """
        if self.center_hit(interval):
            self.s_center.add(interval)
            return self
        else:
            direction = self.hit_branch(interval)
            if not self[direction]:
                self[direction] = InclusiveNode.from_interval(interval)
                self.refresh_balance()
                return self
            else:
                self[direction] = self[direction].add(interval)
                return self.rotate()



