'''
CODER ZERO
connect with me at: https://www.youtube.com/channel/UCKipQAvBc7CWZaPib4y8Ajg
How to train custom yolov5: https://youtu.be/12UoOlsRwh8
DATASET: 1) https://www.kaggle.com/datasets/deepakat002/indian-vehicle-number-plate-yolo-annotation
         2) https://www.kaggle.com/datasets/elysian01/car-number-plate-detection
'''
### importing required libraries
import torch
import cv2
import time
# import pytesseract
import re
import numpy as np
import easyocr
##### DEFINING GLOBAL VARIABLE
EASY_OCR = easyocr.Reader(['en']) ### initiating easyocr
OCR_TH = 0.15




### -------------------------------------- function to run detection ---------------------------------------------------------
def detectx (frame, model):
    frame = [frame]
    print(f"[INFO] Detecting. . . ")
    results = model(frame)
    # results.show()
    # print( results.xyxyn[0])
    # print(results.xyxyn[0][:, -1])
    # print(results.xyxyn[0][:, :-1])

    labels, cordinates = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
    print(cordinates);
    return labels, cordinates

### ------------------------------------ to plot the BBox and results --------------------------------------------------------
def plot_boxes(results, frame,classes):
    """
    --> This function takes results, frame and classes
    --> results: contains labels and coordinates predicted by model on the given frame
    --> classes: contains the strting labels
    """
    labels, cord = results
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]
    print(f"[INFO] Total {n} detections. . . ")
    print(f"[INFO] Looping through all detections. . . ")
    for i in range(n):
        row = cord[i]
        if row[4] >= 0.5: ### threshold value for detection. We are discarding everything below this value
            print(f"[INFO] Extracting BBox coordinates. . . ")
            x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape) ## BBOx coordniates
            text_d = classes[int(labels[i])]
            
            cropped=frame[int(y1):int(y2), int(x1):int(x2)]
            final=imageProcessor(img=cropped)
    
            # final=contours(img=cropped)
            # final = cv2.adaptiveThreshold(final.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 41)
            # # kernel = np.ones((1, 1), np.uint8)
            # opening = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)
            # closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
            # final = cv2.bitwise_or(final, closing)
            # cv2.imshow('final', final)
            cv2.imwrite(("./output/dp.jpg"), final);
            coords = [x1,y1,x2,y2]

            plate_num = recognize_plate_easyocr(img =final, coords= coords, reader= EASY_OCR, region_threshold= OCR_TH)
            print(plate_num)
            # if text_d == 'mask':
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) ## BBox
            cv2.rectangle(frame, (x1, y1-20), (x2, y1), (0, 255,0), -1) ## for text label background
            cv2.putText(frame, f"{plate_num}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255), 2)
            # cv2.imwrite("./output/np.jpg",frame[int(y1)-25:int(y2)+25, int(x1)-25:int(x2)+25])    
    return plate_num



#### ---------------------------- function to recognize license plate --------------------------------------

def imageProcessor(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  

#--- performing Otsu threshold ---
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
    dilation = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rect_kernel);
    ret,thresh1 = cv2.threshold(dilation, 0, 255,cv2.THRESH_OTSU|cv2.THRESH_BINARY)
    # cv2.imshow('thresh1', thresh1)
    dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1)
    # cv2.imshow('dilation', dilation)
    contours, hierarchy = cv2.findContours(dilation,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    im2 = img.copy()
    max1=0
    # img=increase_brightness(im2,-100)
    # cv2.imshow("winname", img)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area=cv2.contourArea(cnt)
        
        if(max1<area):
            xcont,ycont,wcont,hcont=x,y,w,h;
            max1=area
        # cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # cv2.imshow('contours detected', im2);
    # if(max1!=0):
    #     cv2.imshow("Max area",thresh1[ycont:ycont+hcont,xcont:xcont+wcont]);
    # gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    l, a, b = cv2.split(img);

# #-----Applying CLAHE to L-channel-------------------------------------------
    clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    # cv2.imshow('CLAHE output', cl)
            
    limg = cv2.merge((cl,a,b))
#             # cv2.imshow('limg', limg)

# #-----Converting image from LAB Color model to Gray scale model--------------------
    final = cv2.cvtColor(limg, cv2.COLOR_BGR2GRAY)
    final = cv2.fastNlMeansDenoising(final) 
    #final=cv2.adaptiveThreshold(final,200,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2) #imgf contains Binary image
    if(max!=0):    
        return final[ycont:ycont+hcont,xcont:xcont+wcont];
    return final;
# function to recognize license plate numbers using Tesseract OCR
def recognize_plate_easyocr(img, coords,reader,region_threshold):
    # separate coordinates from box
    xmin, ymin, xmax, ymax = coords
    # get the subimage that makes up the bounded region and take an additional 5 pixels on each side
    # nplate = img[int(ymin)-5:int(ymax)+5, int(xmin)-5:int(xmax)+5]
    # nplate = img[int(ymin):int(ymax), int(xmin):int(xmax)] ### cropping the number plate from the whole image
    nplate=img
    ocr_result = reader.readtext(nplate,paragraph=False)
    text = filter_text(region=nplate, ocr_result=ocr_result, region_threshold= region_threshold)
    if len(text) ==1:
        text = text[0].upper()
    return text


### to filter out wrong detections 

def filter_text(region, ocr_result, region_threshold):
    rectangle_size = region.shape[0]*region.shape[1]
    plate = [] 
    print(ocr_result)
    for result in ocr_result:
        length = np.sum(np.subtract(result[0][1], result[0][0]))
        height = np.sum(np.subtract(result[0][2], result[0][1]))
        if length*height / rectangle_size > region_threshold:
            res= ''.join(ch for ch in result[1] if ch.isalnum())  #removing non alphanumeric chars
            plate.append(res)
    return plate


# def increase_brightness(img, value):
#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     h, s, v = cv2.split(hsv)

#     lim = 255 + value
#     print(lim);
#     if lim<=255:
#         v[v < lim] =lim;
#     else:
#         v[v<lim]=255 
    

#     final_hsv = cv2.merge((h, s, v))
#     img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
#     return img


### ---------------------------------------------- Main function -----------------------------------------------------

def main(img_path=None, vid_path=None,vid_out = None):
    print(f"[INFO] Loading model... ");
    ## loading the custom trained model
    # model =  torch.hub.load('ultralytics/yolov5', 'custom', path='last.pt',force_reload=True) ## if you want to download the git repo and then run the detection
    model =  torch.hub.load('./yolov5-master', 'custom', source ='local', path='best.pt',force_reload=True) ### The repo is stored locally
    classes = model.names ### class names in string format
    ### --------------- for detection on image --------------------
    if img_path != None:
        print(f"[INFO] Working with image: {img_path}")
        img_out_name = f"./output/result_{img_path.split('/')[-1]}"
        frame = cv2.imread(img_path) ### reading the image
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        results = detectx(frame, model = model) ### DETECTION HAPPENING HERE    
        frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
        plate_num = plot_boxes(results, frame,classes = classes);
    return plate_num;
        # cv2.namedWindow("img_only", cv2.WINDOW_NORMAL) ## creating a free windown to show the result

        # # while True:
        #     #frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)

        # cv2.imshow("img_only", frame)
        # while cv2.getWindowProperty('img_only', cv2.WND_PROP_VISIBLE) >= 1:
        #     keyCode = cv2.waitKey(20)
        #     if (keyCode & 0xFF) == ord("q"):
        #         cv2.destroyAllWindows()
        #         print(f"[INFO] Exiting. . . ")
        #         cv2.imwrite(f"{img_out_name}",frame)
        #         break
 ## if you want to save he output result.
    ### --------------- for detection on video --------------------
### -------------------  calling the main function------------------------------


# main(vid_path="./test_images/vid_1.mp4",vid_out="vid_1.mp4") ### for custom video
# main(vid_path=0,vid_out="CarTijarat_webcam__result.mp4") #### for webcam

# main(img_path="./test_images/car1.jpg") ## for image
            

