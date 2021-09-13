actions = [
    [
        {
            'type': 'split',
            'args': {
                'mode': 'auto',
                'num': 2,
            }
        },
        {
            'type': 'insert',
            'args': {
                'node': 'random',
                'index': (0, -1),  # (branch_idx, depth)
            }
        }
    ],
] * 7
