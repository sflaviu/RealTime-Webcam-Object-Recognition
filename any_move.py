import cv2
from collections import Counter

from sklearn.externals import joblib
from learning import extract_color_histogram
from learning import image_to_feature_vector

min_area = 2000

labels_so_far = []
nr_reps=0


def diffImg(t0, t1, t2):
    d1 = cv2.absdiff(t2, t1)
    d2 = cv2.absdiff(t1, t0)
    return cv2.bitwise_and(d1, d2)


cam = cv2.VideoCapture(0)

clWinName = "Object movement detection"
bwWinName = "Frame difference in BW"

cv2.namedWindow(bwWinName)
cv2.namedWindow(clWinName)

color_prev = cam.read()[1]
color_curr = cam.read()[1]
color_next = cam.read()[1]

bw_prev = cv2.cvtColor(color_prev, cv2.COLOR_RGB2GRAY)
bw_curr = cv2.cvtColor(color_curr, cv2.COLOR_RGB2GRAY)
bw_next = cv2.cvtColor(color_next, cv2.COLOR_RGB2GRAY)

histo_classifier = joblib.load('model_knn_histo.pkl')
pixel_classifier = joblib.load('model_logistic_pixel.pkl')

classifier=histo_classifier


def most_common(inp):

    c = Counter(inp)

    if c.most_common(1)[0][1]> len(inp) / 2:
       return c.most_common(1)[0][0]
    return "Nothing yet!"

def reset_area(c_area):
    global labels_so_far

    if len(labels_so_far)>15:
        print (most_common(labels_so_far))
    labels_so_far=[]


def find_pixel_object(crop_img):
    features = image_to_feature_vector(crop_img)
    features = [features]

    return pixel_classifier.predict(features)


def find_histo_object(crop_img):
    histo = extract_color_histogram(crop_img)
    histo = [histo]

    return histo_classifier.predict(histo)

feature_method=find_histo_object

def find_object(diff_img, original_img):

    global current_object_area, nr_reps

    thresh = cv2.threshold(diff_img, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)

    _, conts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(conts) > 0:
        c = max(conts, key=cv2.contourArea)

        c_area=cv2.contourArea(c)

        if c_area > min_area:

            if nr_reps>20:

                nr_reps=0
                reset_area(c_area)

            (x, y, w, h) = cv2.boundingRect(c)

            cv2.rectangle(original_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            crop_img = original_img[y:y + h, x:x + w]

            labels_so_far.append(feature_method(crop_img)[0])

            nr_reps=nr_reps+1

    return original_img


def find_movement(diff_img, original_img):
    thresh = cv2.threshold(diff_img, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)

    _, conts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in conts:
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < min_area:
            continue

        # compute the bounding box for the contour, draw it on the frame,
        # and update the text
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(original_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return original_img


process = find_object

while True:

    frame = cam.read()[1]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    color_prev = color_curr
    color_curr = color_next
    color_next = frame

    bw_prev = bw_curr
    bw_curr = bw_next
    bw_next = gray

    bw_difference = diffImg(bw_prev, bw_curr, bw_next)

    processed_frame = process(bw_difference, color_next)

    cv2.imshow(bwWinName, bw_difference)
    cv2.imshow(clWinName, processed_frame)

    key = cv2.waitKey(10)
    if key == 27:
        cv2.destroyWindow(clWinName)
        cv2.destroyWindow(bwWinName)
        break

print("End")
