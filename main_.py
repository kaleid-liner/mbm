import tensorboard_logger as tb_logger
import random


logger = tb_logger.Logger(logdir='./save/refactor/tensorboards/search', flush_secs=2)


best_accs = {
    1: 51.043,
    2: 55.046,
    4: 62.131,
    10: 64.57,
    12: 66.24,
    17: 67.83,
    30: 68.00,
    42: 69.08,
    50: 69.26,
}
best_acc = 0
for i in range(100):
    if i in best_accs:
        best_acc = best_accs[i]
    logger.log_value('best_acc', best_acc, step=i)
