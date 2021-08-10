cd ..
# python main.py --mode bald --query_method bald
# python main.py --mode bald --query_method mean_std
python main.py --mode bald --query_method maxsoftmax
python main.py --mode bald --query_method entropy
python main.py --mode bald --query_method random
python plot.py --mode bald 