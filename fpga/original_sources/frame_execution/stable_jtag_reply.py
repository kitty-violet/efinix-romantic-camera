"""Qualify a held JTAG reply with two matching scans, without resending writes.

The original Bus keeps its address limits, command format and host deadline.
This adapter changes only selector-1 polling; it never issues a command itself.
At the fixed 1 MHz TCK, captures are separated by at least 36 microseconds.
"""
class StableReplyTransport:
    def __init__(self,transport):
        self.transport=transport
        self.poll_pairs=0
        self.unstable_pairs=0

    def scan(self,value,count):
        if (value,count)!=(1,36):return self.transport.scan(value,count)
        first=self.transport.scan(value,count)
        second=self.transport.scan(value,count)
        self.poll_pairs+=1
        # Header TDO bits are not part of the response. A stable error is
        # returned unchanged, so Bus.decode_response still raises it.
        a,b=first>>2,second>>2
        if a&1 and a==b:return second
        if a!=b:self.unstable_pairs+=1
        return 0

    def close(self):return self.transport.close()

