from collections import deque

from ir.tree import Tree
from rl.actor import RemoveActor, InsertActor
from rl.trainer import RLTrainer


def sample_tree(root: Tree, rl_trainer: RLTrainer, random=False):
    # remove stage
    actor_queue = deque()

    for leaf in root.get_leaves():
        actor_queue.append(leaf)

    path = []

    while len(actor_queue) > 0:
        focus_node = actor_queue.pop()

        if random:
            remove_decision = RemoveActor.random_decision()
    
        if remove_decision == 1:
            parent = focus_node.parent
            focus_node.remove()
            if not parent.is_root:
                actor_queue.appendleft(parent)
            path.append(('remove', focus_node))

    # insert stage
    # currently only consider insert after layers
    for leaf in root.get_leaves():
        actor_queue.append(('insert', leaf))

    while len(actor_queue) > 0:
        focus_node = actor_queue.pop()

        if random:
            insert_decision = InsertActor.random_decision()
            compute_type = rl_trainer.get_actor('insert').compute_candidates[insert_decision]
        
        if compute_type != 'Identity':
            new_node = Tree(None, [], compute_type, {}, focus_node.output_channel)
            focus_node.insert(new_node)
            actor_queue.appendleft(new_node)
            path.append(('insert', focus_node, compute_type))
