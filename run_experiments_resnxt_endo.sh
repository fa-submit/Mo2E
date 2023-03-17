#!/usr/bin/env bash

# ENDOSCOPY EXPERIMENTS
CUDA_VISIBLE_DEVICES=1 python3 train_mixup_specialized.py --save_path endo/F1/mxp_1e-1_instance_class/bit_resnext50_1 --csv_train data/train_endo1.csv --data_path data/images --model_name bit_resnext50_1  --do_mixup  0.1    --n_epochs 30 --metric mcc --n_classes 23 --sampling1 'class' --sampling2 'instance' 
CUDA_VISIBLE_DEVICES=1 python3 train_mixup_specialized.py --save_path endo/F1/mxp_2e-1_instance_class/bit_resnext50_1 --csv_train data/train_endo1.csv --data_path data/images --model_name bit_resnext50_1  --do_mixup  0.2    --n_epochs 30 --metric mcc --n_classes 23 --sampling1 'class' --sampling2 'instance'
CUDA_VISIBLE_DEVICES=1 python3 train_mixup_specialized.py --save_path endo/F1/mxp_3e-1_instance_class/bit_resnext50_1 --csv_train data/train_endo1.csv --data_path data/images --model_name bit_resnext50_1  --do_mixup  0.3    --n_epochs 30 --metric mcc --n_classes 23 --sampling1 'class' --sampling2 'instance'

CUDA_VISIBLE_DEVICES=1 python3 train_mixup_specialized.py --save_path endo/F1/mxp_1e-1_instance_instance/bit_resnext50_1 --csv_train data/train_endo1.csv --data_path data/images --model_name bit_resnext50_1  --do_mixup  0.1    --n_epochs 30 --metric mcc --n_classes 23 --sampling1 'instance' --sampling2 'instance' 
CUDA_VISIBLE_DEVICES=1 python3 train_mixup_specialized.py --save_path endo/F1/mxp_2e-1_instance_instance/bit_resnext50_1 --csv_train data/train_endo1.csv --data_path data/images --model_name bit_resnext50_1  --do_mixup  0.2    --n_epochs 30 --metric mcc --n_classes 23 --sampling1 'instance' --sampling2 'instance'
CUDA_VISIBLE_DEVICES=1 python3 train_mixup_specialized.py --save_path endo/F1/mxp_3e-1_instance_instance/bit_resnext50_1 --csv_train data/train_endo1.csv --data_path data/images --model_name bit_resnext50_1  --do_mixup  0.3    --n_epochs 30 --metric mcc --n_classes 23 --sampling1 'instance' --sampling2 'instance'




CUDA_VISIBLE_DEVICES=1 python3 train_mixup_specialized.py --save_path endo/F2/mxp_1e-1_instance_class/bit_resnext50_1 --csv_train data/train_endo2.csv --data_path data/images --model_name bit_resnext50_1  --do_mixup  0.1    --n_epochs 30 --metric mcc --n_classes 23 --sampling1 'class' --sampling2 'instance'
CUDA_VISIBLE_DEVICES=1 python3 train_mixup_specialized.py --save_path endo/F2/mxp_2e-1_instance_class/bit_resnext50_1 --csv_train data/train_endo2.csv --data_path data/images --model_name bit_resnext50_1  --do_mixup  0.2    --n_epochs 30 --metric mcc --n_classes 23 --sampling1 'class' --sampling2 'instance'
CUDA_VISIBLE_DEVICES=1 python3 train_mixup_specialized.py --save_path endo/F2/mxp_3e-1_instance_class/bit_resnext50_1 --csv_train data/train_endo2.csv --data_path data/images --model_name bit_resnext50_1  --do_mixup  0.3    --n_epochs 30 --metric mcc --n_classes 23 --sampling1 'class' --sampling2 'instance'

CUDA_VISIBLE_DEVICES=1 python3 train_mixup_specialized.py --save_path endo/F2/mxp_1e-1_instance_instance/bit_resnext50_1 --csv_train data/train_endo2.csv --data_path data/images --model_name bit_resnext50_1  --do_mixup  0.1    --n_epochs 30 --metric mcc --n_classes 23 --sampling1 'instance' --sampling2 'instance' 
CUDA_VISIBLE_DEVICES=1 python3 train_mixup_specialized.py --save_path endo/F2/mxp_2e-1_instance_instance/bit_resnext50_1 --csv_train data/train_endo2.csv --data_path data/images --model_name bit_resnext50_1  --do_mixup  0.2    --n_epochs 30 --metric mcc --n_classes 23 --sampling1 'instance' --sampling2 'instance'
CUDA_VISIBLE_DEVICES=1 python3 train_mixup_specialized.py --save_path endo/F2/mxp_3e-1_instance_instance/bit_resnext50_1 --csv_train data/train_endo2.csv --data_path data/images --model_name bit_resnext50_1  --do_mixup  0.3    --n_epochs 30 --metric mcc --n_classes 23 --sampling1 'instance' --sampling2 'instance'






CUDA_VISIBLE_DEVICES=1 python3 train_mixup_specialized.py --save_path endo/F3/mxp_1e-1_instance_class/bit_resnext50_1 --csv_train data/train_endo3.csv --data_path data/images --model_name bit_resnext50_1  --do_mixup  0.1    --n_epochs 30 --metric mcc --n_classes 23 --sampling1 'class' --sampling2 'instance'
CUDA_VISIBLE_DEVICES=1 python3 train_mixup_specialized.py --save_path endo/F3/mxp_2e-1_instance_class/bit_resnext50_1 --csv_train data/train_endo3.csv --data_path data/images --model_name bit_resnext50_1  --do_mixup  0.2    --n_epochs 30 --metric mcc --n_classes 23 --sampling1 'class' --sampling2 'instance'
CUDA_VISIBLE_DEVICES=1 python3 train_mixup_specialized.py --save_path endo/F3/mxp_3e-1_instance_class/bit_resnext50_1 --csv_train data/train_endo3.csv --data_path data/images --model_name bit_resnext50_1  --do_mixup  0.3    --n_epochs 30 --metric mcc --n_classes 23 --sampling1 'class' --sampling2 'instance'

CUDA_VISIBLE_DEVICES=1 python3 train_mixup_specialized.py --save_path endo/F3/mxp_1e-1_instance_instance/bit_resnext50_1 --csv_train data/train_endo3.csv --data_path data/images --model_name bit_resnext50_1  --do_mixup  0.1    --n_epochs 30 --metric mcc --n_classes 23 --sampling1 'instance' --sampling2 'instance' 
CUDA_VISIBLE_DEVICES=1 python3 train_mixup_specialized.py --save_path endo/F3/mxp_2e-1_instance_instance/bit_resnext50_1 --csv_train data/train_endo3.csv --data_path data/images --model_name bit_resnext50_1  --do_mixup  0.2    --n_epochs 30 --metric mcc --n_classes 23 --sampling1 'instance' --sampling2 'instance'
CUDA_VISIBLE_DEVICES=1 python3 train_mixup_specialized.py --save_path endo/F3/mxp_3e-1_instance_instance/bit_resnext50_1 --csv_train data/train_endo3.csv --data_path data/images --model_name bit_resnext50_1  --do_mixup  0.3    --n_epochs 30 --metric mcc --n_classes 23 --sampling1 'instance' --sampling2 'instance'



CUDA_VISIBLE_DEVICES=1 python3 train_mixup_specialized.py --save_path endo/F4/mxp_1e-1_instance_class/bit_resnext50_1 --csv_train data/train_endo4.csv --data_path data/images --model_name bit_resnext50_1  --do_mixup  0.1    --n_epochs 30 --metric mcc --n_classes 23 --sampling1 'class' --sampling2 'instance'
CUDA_VISIBLE_DEVICES=1 python3 train_mixup_specialized.py --save_path endo/F4/mxp_2e-1_instance_class/bit_resnext50_1 --csv_train data/train_endo4.csv --data_path data/images --model_name bit_resnext50_1  --do_mixup  0.2    --n_epochs 30 --metric mcc --n_classes 23 --sampling1 'class' --sampling2 'instance'
CUDA_VISIBLE_DEVICES=1 python3 train_mixup_specialized.py --save_path endo/F4/mxp_3e-1_instance_class/bit_resnext50_1 --csv_train data/train_endo4.csv --data_path data/images --model_name bit_resnext50_1  --do_mixup  0.3    --n_epochs 30 --metric mcc --n_classes 23 --sampling1 'class' --sampling2 'instance'

CUDA_VISIBLE_DEVICES=1 python3 train_mixup_specialized.py --save_path endo/F4/mxp_1e-1_instance_instance/bit_resnext50_1 --csv_train data/train_endo4.csv --data_path data/images --model_name bit_resnext50_1  --do_mixup  0.1    --n_epochs 30 --metric mcc --n_classes 23 --sampling1 'instance' --sampling2 'instance' 
CUDA_VISIBLE_DEVICES=1 python3 train_mixup_specialized.py --save_path endo/F4/mxp_2e-1_instance_instance/bit_resnext50_1 --csv_train data/train_endo4.csv --data_path data/images --model_name bit_resnext50_1  --do_mixup  0.2    --n_epochs 30 --metric mcc --n_classes 23 --sampling1 'instance' --sampling2 'instance'
CUDA_VISIBLE_DEVICES=1 python3 train_mixup_specialized.py --save_path endo/F4/mxp_3e-1_instance_instance/bit_resnext50_1 --csv_train data/train_endo4.csv --data_path data/images --model_name bit_resnext50_1  --do_mixup  0.3    --n_epochs 30 --metric mcc --n_classes 23 --sampling1 'instance' --sampling2 'instance'
