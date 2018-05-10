#! /bin/bash -e

# echo "filter html"

cd data/

python ../code/tools.py -m filter -s News_info_train_example100.txt -t News_info_train_example100_filter.txt
python ../code/tools.py -m filter -s News_info_train.txt -t News_info_train_filter.txt
python ../code/tools.py -m filter -s News_info_validate.txt -t News_info_validate_filter.txt
