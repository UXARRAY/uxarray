from intervaltree import IntervalTree
from .inclusive_node import InclusiveNode
from .inclusive_interval import InclusiveInterval
from sortedcontainers import SortedDict
from copy import copy

class InclusiveIntervalTree(IntervalTree):
    def __init__(self, inclusive_intervals=None):
        """
        Set up the all-inclusive tree by hacking the boundary table.

        Completes in O(n*log n) time.
        """
        inclusive_intervals = set(inclusive_intervals) if inclusive_intervals is not None else set()
        for iv in inclusive_intervals:
            if iv.is_null():
                raise ValueError(
                    "IntervalTree: Null Interval objects not allowed in IntervalTree:"
                    " {0}".format(iv)
                )
        self.all_intervals = inclusive_intervals
        self.top_node = InclusiveNode.from_intervals(self.all_intervals)
        self.boundary_table = SortedDict()
        for iv in self.all_intervals:
            self._add_boundaries(iv)

    @classmethod
    def from_tuples(cls, tups):
        """
        Create a new IntervalTree from an iterable of 2- or 3-tuples,
         where the tuple lists begin, end, and optionally data.
        """
        ivs = [InclusiveInterval(*t) for t in tups]
        return InclusiveIntervalTree(ivs)

    def copy(self):
        """
        Construct a new IntervalTree using shallow copies of the
        intervals in the source tree.

        Completes in O(n*log n) time.
        :rtype: InclusiveIntervalTree
        """
        return InclusiveIntervalTree(iv.copy() for iv in self)

    def add(self, interval):
        """
        Adds an interval to the tree, if not already present.

        Completes in O(log n) time.
        """
        if interval in self:
            return

        if interval.is_null():
            raise ValueError(
                "IntervalTree: Null Interval objects not allowed in IntervalTree:"
                " {0}".format(interval)
            )

        if not self.top_node:
            self.top_node = InclusiveNode.from_interval(interval)
        else:
            self.top_node = self.top_node.add(interval)
        self.all_intervals.add(interval)
        self._add_boundaries(interval)
    append = add

    def at(self, p):
        """
        Returns the set of all intervals that contain p.

        Completes in O(m + log n) time, where:
          * n = size of the tree
          * m = number of matches
        :rtype: set of Interval
        """
        root = self.top_node
        if not root:
            return set()
        return root.search_point(p, set())

    def addi(self, begin, end, data=None):
        """
        Shortcut for add(Interval(begin, end, data)).

        Completes in O(log n) time.
        """
        return self.add(InclusiveInterval(begin, end, data))
    appendi = addi

    def range(self):
        """
        Returns a minimum-spanning Interval that encloses all the
        members of this IntervalTree. If the tree is empty, returns
        null Interval.
        :rtype: Interval
        """
        return InclusiveInterval(self.begin(), self.end())

    def removei(self, begin, end, data=None):
        """
        Shortcut for remove(Interval(begin, end, data)).

        Completes in O(log n) time.
        """
        return self.remove(InclusiveInterval(begin, end, data))

    def discardi(self, begin, end, data=None):
        """
        Shortcut for discard(Interval(begin, end, data)).

        Completes in O(log n) time.
        """
        return self.discard(InclusiveInterval(begin, end, data))

    def difference(self, other):
        """
        Returns a new tree, comprising all intervals in self but not
        in other.
        """
        ivs = set()
        for iv in self:
            if iv not in other:
                ivs.add(iv)
        return InclusiveIntervalTree(ivs)

    def union(self, other):
        """
        Returns a new tree, comprising all intervals from self
        and other.
        """
        return InclusiveIntervalTree(set(self).union(other))

    def intersection(self, other):
        """
        Returns a new tree of all intervals common to both self and
        other.
        """
        ivs = set()
        shorter, longer = sorted([self, other], key=len)
        for iv in shorter:
            if iv in longer:
                ivs.add(iv)
        return InclusiveIntervalTree(ivs)

    def symmetric_difference(self, other):
        """
        Return a tree with elements only in self or other but not
        both.
        """
        if not isinstance(other, set): other = set(other)
        me = set(self)
        ivs = me.difference(other).union(other.difference(me))
        return InclusiveIntervalTree(ivs)

    def chop(self, begin, end, datafunc=None):
        """
        Like remove_envelop(), but trims back Intervals hanging into
        the chopped area so that nothing overlaps.
        """
        insertions = set()
        begin_hits = [iv for iv in self.at(begin) if iv.begin < begin]
        end_hits = [iv for iv in self.at(end) if iv.end > end]

        if datafunc:
            for iv in begin_hits:
                insertions.add(InclusiveInterval(iv.begin, begin, datafunc(iv, True)))
            for iv in end_hits:
                insertions.add(InclusiveInterval(end, iv.end, datafunc(iv, False)))
        else:
            for iv in begin_hits:
                insertions.add(InclusiveInterval(iv.begin, begin, iv.data))
            for iv in end_hits:
                insertions.add(InclusiveInterval(end, iv.end, iv.data))

        self.remove_envelop(begin, end)
        self.difference_update(begin_hits)
        self.difference_update(end_hits)
        self.update(insertions)

    def slice(self, point, datafunc=None):
        """
        Split Intervals that overlap point into two new Intervals. if
        specified, uses datafunc(interval, islower=True/False) to
        set the data field of the new Intervals.
        :param point: where to slice
        :param datafunc(interval, isupper): callable returning a new
        value for the interval's data field
        """
        hitlist = set(iv for iv in self.at(point) if iv.begin < point)
        insertions = set()
        if datafunc:
            for iv in hitlist:
                insertions.add(InclusiveInterval(iv.begin, point, datafunc(iv, True)))
                insertions.add(InclusiveInterval(point, iv.end, datafunc(iv, False)))
        else:
            for iv in hitlist:
                insertions.add(InclusiveInterval(iv.begin, point, iv.data))
                insertions.add(InclusiveInterval(point, iv.end, iv.data))
        self.difference_update(hitlist)
        self.update(insertions)

    def find_nested(self):
        """
        Returns a dictionary mapping parent intervals to sets of
        intervals overlapped by and contained in the parent.

        Completes in O(n^2) time.
        :rtype: dict of [Interval, set of Interval]
        """
        result = {}

        def add_if_nested():
            if parent.contains_interval(child):
                if parent not in result:
                    result[parent] = set()
                result[parent].add(child)

        long_ivs = sorted(self.all_intervals, key=InclusiveInterval.length, reverse=True)
        for i, parent in enumerate(long_ivs):
            for child in long_ivs[i + 1:]:
                add_if_nested()
        return result

    def split_overlaps(self):
        """
        Finds all intervals with overlapping ranges and splits them
        along the range boundaries.

        Completes in worst-case O(n^2*log n) time (many interval
        boundaries are inside many intervals), best-case O(n*log n)
        time (small number of overlaps << n per interval).
        """
        if not self:
            return
        if len(self.boundary_table) == 2:
            return

        bounds = sorted(self.boundary_table)  # get bound locations

        new_ivs = set()
        for lbound, ubound in zip(bounds[:-1], bounds[1:]):
            for iv in self[lbound]:
                new_ivs.add(InclusiveInterval(lbound, ubound, iv.data))

        self.__init__(new_ivs)

    def merge_overlaps(self, data_reducer=None, data_initializer=None, strict=True):
        """
        Finds all intervals with overlapping ranges and merges them
        into a single interval. If provided, uses data_reducer and
        data_initializer with similar semantics to Python's built-in
        reduce(reducer_func[, initializer]), as follows:

        If data_reducer is set to a function, combines the data
        fields of the Intervals with
            current_reduced_data = data_reducer(current_reduced_data, new_data)
        If data_reducer is None, the merged Interval's data
        field will be set to None, ignoring all the data fields
        of the merged Intervals.

        On encountering the first Interval to merge, if
        data_initializer is None (default), uses the first
        Interval's data field as the first value for
        current_reduced_data. If data_initializer is not None,
        current_reduced_data is set to a shallow copy of
        data_initializer created with copy.copy(data_initializer).

        If strict is True (default), intervals are only merged if
        their ranges actually overlap; adjacent, touching intervals
        will not be merged. If strict is False, intervals are merged
        even if they are only end-to-end adjacent.

        Completes in O(n*logn).
        """
        if not self:
            return

        sorted_intervals = sorted(self.all_intervals)  # get sorted intervals
        merged = []
        # use mutable object to allow new_series() to modify it
        current_reduced = [None]
        higher = None  # iterating variable, which new_series() needs access to

        def new_series():
            if data_initializer is None:
                current_reduced[0] = higher.data
                merged.append(higher)
                return
            else:  # data_initializer is not None
                current_reduced[0] = copy(data_initializer)
                current_reduced[0] = data_reducer(current_reduced[0], higher.data)
                merged.append(InclusiveInterval(higher.begin, higher.end, current_reduced[0]))

        for higher in sorted_intervals:
            if merged:  # series already begun
                lower = merged[-1]
                if (higher.begin < lower.end or
                        not strict and higher.begin == lower.end):  # should merge
                    upper_bound = max(lower.end, higher.end)
                    if data_reducer is not None:
                        current_reduced[0] = data_reducer(current_reduced[0], higher.data)
                    else:  # annihilate the data, since we don't know how to merge it
                        current_reduced[0] = None
                    merged[-1] = InclusiveInterval(lower.begin, upper_bound, current_reduced[0])
                else:
                    new_series()
            else:  # not merged; is first of Intervals to merge
                new_series()

        self.__init__(merged)

    def containsi(self, begin, end, data=None):
        """
        Shortcut for (Interval(begin, end, data) in tree).

        Completes in O(1) time.
        :rtype: bool
        """
        return InclusiveInterval(begin, end, data) in self

    def __reduce__(self):
        """
        For pickle-ing.
        :rtype: tuple
        """
        return InclusiveIntervalTree, (sorted(self.all_intervals),)


