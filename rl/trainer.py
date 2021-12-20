import torch.nn as nn
from .actor import InsertActor, RemoveActor


class RLTrainer:
    def __init__(self, compute_candidates, encoder_hidden_size, actor_config):
        self.actors = nn.ModuleList([
            InsertActor(compute_candidates, encoder_hidden_size, actor_config),
            RemoveActor(encoder_hidden_size, actor_config),
        ])

    def get_actor(self, actor_type):
        if actor_type == 'insert':
            return self.actors[0]
        else:
            return self.actors[1]