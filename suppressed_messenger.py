class SuppressedMessenger(object):
    """ A class to output messages with output being supressed at a certain point."""
    def __init__(self,name,max_messages):
        self.name = name
        self.num_printed = 0
        self.max_messages = max_messages
        self.stopped_printing = False if max_messages > 0 else True

    def print(self,msg):
        if self.num_printed < self.max_messages:
            print(msg)
            self.num_printed += 1
        elif not self.stopped_printing:
            print("[Further output regarding "+self.name+" will be suppressed]")
            self.stopped_printing = True
