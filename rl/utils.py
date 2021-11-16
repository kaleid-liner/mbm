import torch


def lstm_zero_hidden_state(hidden_size):
    h0 = torch.zeros(1, hidden_size, requires_grad=True)
    c0 = torch.zeros(1, hidden_size, requires_grad=True)
    return h0, c0  # [1, hidden_size]
