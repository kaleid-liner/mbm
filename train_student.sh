# python main.py --learning_rate 0.04 --ckpt ./save/student_init_models/fsp_init_30.pth --tb_path ./save/tensorboards/student --train_student
python main.py --learning_rate 0.0008 --ckpt ./save/teacher_stride2/mobilenetv2_best.pth --tb_path ./save/tensorboards/student_larger_stride2_noinsert --train_student --save_folder ./save/student_larger_stride2_noinsert --lr_scheduler reduce
