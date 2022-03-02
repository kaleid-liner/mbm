from nn_meter import load_latency_predictor

import math

from ir.tree import Tree


device_map = {
    'cpu': 'cortexA76cpu_tflite21',
    'gpu': 'adreno640gpu_tflite21',
    'vpu': 'myriadvpu_openvino2019r2',
}


def get_latency(model, hw=224, stem_stride=1, main_device='gpu'):
    """
    workaround
    """
    if main_device == 'cpu':
        devices = [
            'cpu',
            'gpu',
        ]
    else:
        devices = [
            'gpu',
            'cpu',
        ]
    predictors = [load_latency_predictor(device_map[device]) for device in devices]

    hw = hw // stem_stride

    input_shapes = [
        (1, 32, hw // 1, hw // 1),
        (1, 16, hw // 1, hw // 1),
        (1, 24, hw // 2, hw // 2),
        (1, 32, hw // 4, hw // 4),
        (1, 64, hw // 8, hw // 8),
        (1, 96, hw // 8, hw // 8),
        (1, 160, hw // 16, hw // 16),
    ]

    total_lat = 0
    for input_shape, block in zip(input_shapes, model.blocks):
        left_branch_lat = predictors[0].predict(block.branches[0], 'torch', input_shape=input_shape)
        print('Branch 0 on {}: {}'.format(devices[0], left_branch_lat))
        if len(block.branches) > 1:
            right_branch_lat = predictors[1].predict(block.branches[1], 'torch', input_shape=input_shape)
            print('Branch 1 on {}: {}'.format(devices[1], right_branch_lat))
            total_lat += max(left_branch_lat, right_branch_lat)
        else:
            total_lat += left_branch_lat

    return total_lat


def process_trees_manually(trees, model):
    if model == 'mobilenetv2':
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1, 1],
            [6, 24, 2, 2, 2],
            [6, 32, 3, 2, 2],
            [6, 64, 4, 2, 2],
            [6, 96, 3, 1, 2],
            [6, 160, 3, 2, 2],
            [6, 320, 1, 1, 1],
        ]

        for i, setting, tree in zip(range(7), inverted_residual_setting, trees):
            t, c, n, s, f = setting
            if f != 1:
                right_leaf, left_leaf = tree.get_leaves()
                if n == 2:
                    left_leaf.remove()
                if n == 3:
                    if i == 4:
                        left_leaf.remove()
                        right_leaf.remove()
                        # right_leaf.parent.remove()
                    else:
                        right_leaf.remove()
                        left_leaf.remove()
                if n == 4:
                    right_leaf.remove()
                    left_leaf.remove()
                    left_leaf.parent.remove()

                right_leaf, left_leaf = tree.get_leaves()
                right_leaf.insert(Tree(None, [], 'conv_5x5', {}, right_leaf.output_channel))
                right_leaf = right_leaf.children[0]
                right_leaf.insert(Tree(None, [], 'conv_1x1', {}, right_leaf.output_channel))

                left_leaf.insert(Tree(None, [], 'conv_3x3', {}, left_leaf.output_channel))
                left_leaf = left_leaf.children[0]
                left_leaf.insert(Tree(None, [], 'conv_1x1', {}, left_leaf.output_channel))
    elif model == 'resnet20':
        block_settings = [
            # c, n, s, f
            [16, 3, 1, 2],
            [32, 3, 2, 2],
            [64, 3, 2, 2]
        ]
        for setting, tree in zip(block_settings, trees):
            right_leaf, left_leaf = tree.get_leaves()
            right_leaf.remove()
            left_leaf.remove()
            left_leaf.parent.remove()

            right_leaf, left_leaf = tree.get_leaves()
            right_leaf.insert(Tree(None, [], 'conv_1x1', {}, right_leaf.output_channel))
            right_leaf = right_leaf.children[0]
            right_leaf.insert(Tree(None, [], 'dwconv_5x5', {}, right_leaf.output_channel))

            left_leaf.insert(Tree(None, [], 'conv_1x1', {}, left_leaf.output_channel))
            left_leaf = left_leaf.children[0]
            left_leaf.insert(Tree(None, [], 'conv_3x3', {}, left_leaf.output_channel))

    elif model == 'resnet32':
        block_settings = [
            # c, n, s, f
            [16, 5, 1, 2],
            [32, 5, 2, 2],
            [64, 5, 2, 2]
        ]
        for setting, tree in zip(block_settings, trees):
            right_leaf, left_leaf = tree.get_leaves()
            right_leaf.remove()
            right_leaf.parent.remove()
            left_leaf.remove()
            left_leaf.parent.remove()
            left_leaf.parent.parent.remove()

            right_leaf, left_leaf = tree.get_leaves()
            right_leaf.insert(Tree(None, [], 'conv_1x1', {}, right_leaf.output_channel))
            right_leaf = right_leaf.children[0]
            right_leaf.insert(Tree(None, [], 'dwconv_5x5', {}, right_leaf.output_channel))
            right_leaf = right_leaf.children[0]
            right_leaf.insert(Tree(None, [], 'conv_3x3', {}, right_leaf.output_channel))

            left_leaf.insert(Tree(None, [], 'conv_1x1', {}, left_leaf.output_channel))
            left_leaf = left_leaf.children[0]
            left_leaf.insert(Tree(None, [], 'conv_3x3', {}, left_leaf.output_channel))

    elif model == 'vgg':
        block_settings = [
            # c, n, f
            [64, 2, 2],
            [128, 2, 2],
            [256, 3, 2],
            [512, 3, 2],
            [512, 3, 2],
        ]
        for setting, tree in zip(block_settings, trees):
            c, n, f = setting
            if n == 2:
                right_leaf, left_leaf = tree.get_leaves()
                right_leaf.remove()
            elif n == 3:
                right_leaf, left_leaf = tree.get_leaves()
                left_leaf.remove()
                right_leaf.remove()
                right_leaf.parent.remove()

            right_leaf, left_leaf = tree.get_leaves()
            right_leaf.insert(Tree(None, [], 'conv_1x1', {}, right_leaf.output_channel))
            right_leaf = right_leaf.children[0]
            right_leaf.insert(Tree(None, [], 'dwconv_3x3', {}, right_leaf.output_channel))

    return trees
