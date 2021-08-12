cd ..
python main.py --mode bald --query_method bald --init_dataset 200 --query_step 20 --query_size 100 --epoch 200 --id 2
python main.py --mode bald --query_method mean_std --init_dataset 200 --query_step 20 --query_size 100 --epoch 200 --id 2
python main.py --mode bald --query_method maxsoftmax --init_dataset 200 --query_step 20 --query_size 100 --epoch 200 --id 2
python main.py --mode bald --query_method entropy --init_dataset 200 --query_step 20 --query_size 100 --epoch 200 --id 2
python main.py --mode bald --query_method random --init_dataset 200 --query_step 20 --query_size 100 --epoch 200 --id 2
python plot.py --mode bald --id 2