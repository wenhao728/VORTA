import os

import torch
import torch.distributed as dist


class SequenceParallelState:
    def __init__(self):
        self._enabled: bool = False
        self._sp_size: int = 1
        self._group_id: int = 0
        self._group_local_rank: int = 0
        self._group = None

    @property
    def rank(self):
        return int(os.getenv("RANK", "0"))

    @property
    def local_rank(self):
        return int(os.getenv("LOCAL_RANK", "0"))

    @property
    def world_size(self):
        return int(os.getenv("WORLD_SIZE", "1"))
    
    @property
    def enabled(self):
        return self._enabled
    
    @property
    def sp_size(self):
        return self._sp_size
    
    @property
    def group_id(self):
        return self._group_id
    
    @property
    def group_local_rank(self):
        return self._group_local_rank

    @property
    def group(self):
        return self._group
    
    @property
    def num_sp_groups(self):
        return self.world_size // self.sp_size

    def cleanup(self):
        dist.destroy_process_group()
        # self.setup_sp_group(sequence_parallel_size=1)

    def setup_sp_group(self, sequence_parallel_size: int):
        if self.world_size % sequence_parallel_size != 0:
            raise ValueError(f"{self.world_size=} must be divisible by {sequence_parallel_size=}!")
        
        if sequence_parallel_size > 1:
            self._enabled = True
            self._sp_size = sequence_parallel_size
            self._group_id = self.rank // sequence_parallel_size
            self._group_local_rank = self.rank % sequence_parallel_size

            ranks = list(range(self._group_id * sequence_parallel_size, (self._group_id + 1) * sequence_parallel_size))
            self._group = dist.new_group(ranks)
        else:
            self._enabled = False
            self._sp_size = 1
            self._group_id = self.rank
            self._group_local_rank = 0
            self._group = None


SP_STATE = SequenceParallelState()