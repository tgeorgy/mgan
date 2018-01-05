mkdir -p data
mkdir -p data/celeba
mkdir -p data/celeba/pos
mkdir -p data/celeba/neg

if [ ! -f data/img_align_celeba.zip ]; then
    echo "img_align_celeba.zip not found!"
fi

if [ ! -f data/list_attr_celeba.zip ]; then
    echo "list_attr_celeba.txt not found!"
fi

cd data
unzip img_align_celeba.zip

cd ..
python crop_images.py
