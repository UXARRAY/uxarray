from intervaltree.interval import Interval


class InclusiveInterval(Interval):
    """
    An interval with inclusive boundaries.
    """

    def overlaps(self, begin, end=None):
        """
        Make it inclusive for the beginning and end.
        """
        if end is not None:
            # An overlap means that some C exists that is inside both ranges:
            #   begin <= C <= end
            # and
            #   self.begin <= C <= self.end
            # See https://stackoverflow.com/questions/3269434/whats-the-most-efficient-way-to-test-two-integer-ranges-for-overlap/3269471#3269471
            return begin <= self.end and end >= self.begin
        try:
            return self.overlaps(begin.begin, begin.end)
        except:
            return self.contains_point(begin)

    def contains_point(self, p):
        """
        Make it inclusive for the beginning and end.
        """
        return self.begin <= p <= self.end

