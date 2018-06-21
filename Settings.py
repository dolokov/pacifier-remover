import os 

directory_base = '/data/pacifier/face_crops'
directory_unlabeled = os.path.join(directory_base,'unlabeled')
directory_labeled = os.path.join(directory_base,'labeled')
directory_pacifier = os.path.join(directory_labeled,'with')
directory_wo_pacifier = os.path.join(directory_labeled,'without')
directory_rejected = os.path.join(directory_labeled,'rejected')

file_extensions = ['.jpg','.jpeg','.png','.bmp']
