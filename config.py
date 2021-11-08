split_actions = [
    {
        'type': 'split',
        'args': {
            'mode': 'full',
            'num': 2,
        }
    },
]

actions = [split_actions] + [
    split_actions + [
        {
            'type': 'insert',
            'args': {
                'node': 'conv_1x1',
                'index': (0, 999),
            }
        },
        {
            'type': 'insert',
            'args': {
                'node': 'dwconv_7x7',
                'index': (0, 999),  # (branch_idx, depth)
            }
        },
        {
            'type': 'insert',
            'args': {
                'node': 'conv_1x1',
                'index': (1, 999),
            }
        },
        {
            'type': 'insert',
            'args': {
                'node': 'dwconv_3x3',
                'index': (1, 999),
            }
        },
    ]
] * 5 + [split_actions]
