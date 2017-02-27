import cv2
import os
import h5py
import numpy as np

ipath = 'data/img_align_celeba/'

selected_attr = 'Smiling'

crop = [50, 10, 10, 10]  # top, bottom, left, right
target_size = (128, 128)  # w, h
print_every = 2000

nimages = len([f for f in os.listdir(ipath) if '.jpg' in f])

if not os.path.exists('cache'):
    os.mkdir('cache')

with h5py.File('cache/train.h5', 'w') as hf:
    hf.create_dataset('img', (nimages, 3)+target_size, dtype='uint8')
    hf.create_dataset('attrs', (nimages, 1), dtype='uint8')

with open('data/list_attr_celeba.txt') as f:
    attrs = [l for l in f]

header = attrs[1].split(' ')
header = {h: i for i, h in enumerate(header)}
assert selected_attr in header, 'check attribute name'
col_id = header[selected_attr] + 1

acc = 0
with h5py.File('cache/train.h5') as hf:
    hf['attr_name'] = selected_attr

    for row in attrs[2:]:
        cols = row.split(' ')
        cols = [c for c in cols if c is not '']

        fname = cols[0]
        fpath = os.path.join(ipath, fname)
        if os.path.exists(fpath):
            fpath = os.path.join(ipath, fname)
            img = cv2.imread(fpath)
            img = img[crop[0]:img.shape[0]-crop[1],
                      crop[2]:img.shape[1]-crop[3],
                      ::-1]
            img = cv2.resize(img, target_size)
            img = np.swapaxes(img, 0, 2)
            img = np.swapaxes(img, 1, 2)

            hf['img'][acc] = img

            attr = cols[col_id]
            hf['attrs'][acc] = attr == '1'

            acc += 1
            if acc % print_every == 0:
                print 'saved %d images' % acc
