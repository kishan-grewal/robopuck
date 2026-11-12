import torch

class Timer:
    # start record must go in __enter__ no matter what, never in __init__

    # def __init__(self):
    #     self.start = torch.cuda.Event(enable_timing=True)
    #     self.end = torch.cuda.Event(enable_timing=True)
    def __enter__(self):
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        # sync, start and give self
        torch.cuda.synchronize()
        self.start.record()
        return self
    def __exit__(self, exc_type, exc, tb):
        # end, sync and find elapsed
        self.end.record()
        torch.cuda.synchronize()
        self.ms = self.start.elapsed_time(self.end)