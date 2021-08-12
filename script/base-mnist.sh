cd ..
python main.py --mode base --query_method coreset --init_dataset 200 --query_step 20 --query_size 100 --epoch 200 --id 2 --gpu 1
python main.py --mode base --query_method maxsoftmax --init_dataset 200 --query_step 20 --query_size 100 --epoch 200 --id 2 --gpu 1
python main.py --mode base --query_method entropy --init_dataset 200 --query_step 20 --query_size 100 --epoch 200 --id 2 --gpu 1
python main.py --mode base --query_method random --init_dataset 200 --query_step 20 --query_size 100 --epoch 200 --id 2 --gpu 1
python plot.py --id 2 0 --mode base