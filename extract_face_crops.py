from __future__ import print_function

""" 
this offers getBoundingBoxes which extracts bounding boxes for all faces 

http://docs.opencv.org/trunk/d7/d8b/tutorial_py_face_detection.html

for better quality reimplement https://arxiv.org/pdf/1502.02766v3.pdf

python pacifier-remover/extract_face_crops.py --input /data/pacifier/images --output /data/pacifier/face_crops/unlabeled
"""

import numpy as np
import cv2
import dlib

class FaceDetector(object):
    def __init__(self):
        #fn_xml_face = '/home/deeplearning/python-libs/opencv/data/haarcascades/haarcascade_frontalface_default.xml'
        fn_xml_face = 'pacifier-remover/haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(fn_xml_face)      

        # initialize dlib's face detector (HOG-based) and then create
        # the facial landmark predictor
        self.detector = dlib.get_frontal_face_detector()
        try:
            self.predictor = dlib.shape_predictor('/data/pacifier/shape_predictor_68_face_landmarks.dat')
        except:
            print(""" could not find shape predictor. please download from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 """)
            raise Exception()

    def getBoundingBoxescv2(self,image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
         # create a CLAHE object (Arguments are optional).
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        return faces 

    def getBoundingBoxesdlib(self,image):
        #https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.shape[-1]==3 else image
        multiscale_passes = 1
        faces = self.detector(gray, multiscale_passes)

        # convert ((x,y),(x+w,y+h)) => (x,y,w,h)
        def rect_to_bb(rect):
            # take a bounding predicted by dlib and convert it
            # to the format (x, y, w, h) as we would normally do
            # with OpenCV
            x = rect.left()
            y = rect.top()
            w = rect.right() - x
            h = rect.bottom() - y
         
            # return a tuple of (x, y, w, h)
            return (x, y, w, h)

        faces = [ rect_to_bb(face) for face in faces]
        return faces

    def getBoundingBoxes(self,image):
        bboxes =  self.getBoundingBoxesdlib(image)
        if not bboxes:
            bboxes = self.getBoundingBoxescv2(image)
        #print('[*] detected',bboxes)
        return bboxes

    def getCrops(self,image):
        return [ image[y:y+h,x:x+w] for (x,y,w,h) in self.getBoundingBoxes(image) ]
        
    def getDrawnBoundingBoxes(self,image):
        if len(image.shape) == 3:
            color = (255,0,np.random.randint(0,255))
        else:
            color = np.random.randint(0,255)

        thick = int(round(min(image.shape[:2]) * 0.01))
        for (x,y,w,h) in self.getBoundingBoxes(image):
            cv2.rectangle(image,(x,y),(x+w,y+h),color,thick)

        return image


def main():
    from glob import glob 
    import os
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--drawBoundingBox',type=bool,default=False)
    args = parser.parse_args()
    fn_image_input = args.input 
    fn_image_output = args.output

    if os.path.isdir(fn_image_input):
        fns_in = glob(os.path.join(fn_image_input,'*'))
        if not os.path.isdir(fn_image_output):#,'Output should be directory if input is directory'
            os.makedirs(fn_image_output)
        fns_out = [ os.path.join(fn_image_output,fn_in.split('/')[-1]) for fn_in in fns_in ]
    else:
        fns_in = [fn_image_input]
        fns_out = [fn_image_output]

    face_detector = FaceDetector()

    if args.drawBoundingBox:
        for i,fn_image_input in enumerate(fns_in):
            try:
                img = cv2.imread(fn_image_input)
                if img is not None: 
                    img = face_detector.getDrawnBoundingBoxes(img)
                    cv2.imwrite(fns_out[i],img)
                    if i % int(len(fns_in)/10):
                        print('[*] drawBoundingBox %i/%i'%(i,len(fns_in)))
                else:
                    print('[WARNING]',fn_image_input)
            except:
                ''
        return

    ## do face crops
    for i,fn_image_input in enumerate(fns_in):
        fext = '.%s'%fn_image_input.split('.')[-1]
        img = cv2.imread(fn_image_input)
        if img is not None:
            crops = face_detector.getCrops(img)
            for j,crop in enumerate(crops):
                cfn = fns_out[i].replace(fext,'%i%s'%(j,fext))
                try:
                    cv2.imwrite(cfn,crop)
                except:
                    ''
            print('[*] %i/%i'%(i,len(fns_in)))

if __name__ == '__main__':
    main()