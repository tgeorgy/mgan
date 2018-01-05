import os
import json

ipath = 'data/img_align_celeba/'
split_by = 'Smiling'

with open('data/list_attr_celeba.txt') as f:
    attrs = [l for l in f]

attr_id = 0
for att_name in attrs[1].split(' '):
    attr_id += 1
    if att_name == split_by:
        break

info = {}
for l in attrs[2:]:
    lbls = l.replace('  ', ' ').split(' ')
    info[lbls[0]] = lbls[attr_id] == '1'

for fname in os.listdir(ipath):
    if info[fname]:
        os.rename(ipath+fname, 'data/celeba/pos/'+fname)
    else:
        os.rename(ipath+fname, 'data/celeba/neg/'+fname)
