# this file is for copying UHDM train images to a single directory

import os
import shutil

images = []
dirs = os.listdir('/datapath/UHDM/train')
print(dirs)
for d in dirs:
    for _, _, imgs in sorted(os.walk(os.path.join('/datapath/UHDM/train', d))):
         for img in imgs:
             img_path = os.path.join('/datapath/UHDM/train', d, img)
             if 'gt' in img_path:
                 shutil.copy(img_path, os.path.join('/datapath/UHDM/train_all/clear', d+img))
             else:
                 shutil.copy(img_path, os.path.join('/datapath/UHDM/train_all/moire', d+img))
             
    
#for dirpath, dirs, fnames in sorted(os.walk('../train')):
#    for 
