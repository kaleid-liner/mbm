# python main.py --learning_rate 0.04 --ckpt ./save/student_init_models/fsp_init_30.pth --tb_path ./save/tensorboards/student --train_student
python main.py --learning_rate 0.1 --tb_path ./save/refactor/tensorboards/teacher --save_folder ./save/refactor/models/teacher --lr_scheduler multistep --device 0
