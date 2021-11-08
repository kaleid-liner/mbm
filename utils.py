from nn_meter import load_latency_predictor


device_map = {
    'cpu': 'cortexA76cpu_tflite21',
    'gpu': 'adreno640gpu_tflite21',
    'vpu': 'myriadvpu_openvino2019r2',
}


def get_latency(model, device='cpu', hw=224, stem_stride=1, branch=True):
    """
    workaround
    """
    predictor = load_latency_predictor(device_map[device])
    total_lat = predictor.predict(model, 'torch', input_shape=(1, 3, hw, hw))
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
    if branch:
        for input_shape, block in zip(input_shapes, model.blocks):
            if len(block.branches) > 1:
                branch_lat = predictor.predict(block.branches[0], 'torch', input_shape=input_shape)
                total_lat -= branch_lat

    return total_lat
