python main.py --learning_rate 0.008 --ckpt /data/workspace/wjiany/pretrained/cifar100_vgg16_bn-7d8c4031.pt --tb_path ./save/vgg/tensorboards/student_0 --train_student --save_folder ./save/vgg/models/student_0 --lr_scheduler reduce --device 0 --dataset cifar100 --model vgg --parallel