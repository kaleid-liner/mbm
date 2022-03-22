torchrun --nproc_per_node=8 distributed_train.py --save_folder ./save/resnet50 --model resnet50 --data_folder /data/workspace/wjiany/ILSVRC/Data/CLS-LOC --batch_size 32 --weight_decay 0.00001 --distill --train_epoch 120
torchrun --nproc_per_node=8 distributed_train.py --save_folder ./save/mobilenetv2/train_distill --model imagenet_mobilenetv2 --data_folder /data/workspace/wjiany/ILSVRC/Data/CLS-LOC --batch_size 32 --train_epoch 300 --learning_rate 0.045 --weight_decay 0.00004 --lr_step_size 1 --lr_gamma 0.98 --distill
torchrun --nproc_per_node=8 distributed_train.py --save_folder ./save/mobilenetv2/train_without_distill --model imagenet_mobilenetv2 --data_folder /data/workspace/wjiany/ILSVRC/Data/CLS-LOC --batch_size 32 --train_epoch 300 --learning_rate 0.045 --weight_decay 0.00004 --lr_step_size 1 --lr_gamma 0.98