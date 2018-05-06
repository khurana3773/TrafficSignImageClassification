from flask import Flask, render_template ,request, jsonify
import cv2
import os
import numpy as np

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__,template_folder='templates')


@app.route('/')
def main():
    return render_template('hello.html', name='Suraj')




def processImage(file):
    bgr = [30, 30, 200]
    thresh = 40
    # bright = cv2.imread('C:/Users/Suraj/PycharmProjects/255/Bright.png')
    # dark = cv2.imread(file)

    bright = cv2.imread(file.filename)

    bright2 = cv2.imread(file.filename)

    minBGR = np.array([bgr[0] - thresh, bgr[1] - thresh, bgr[2] - thresh])
    maxBGR = np.array([bgr[0] + thresh, bgr[1] + thresh, bgr[2] + thresh])

    maskBGR = cv2.inRange(bright, minBGR, maxBGR)
    resultBGR = cv2.bitwise_and(bright, bright, mask=maskBGR)
    ret, threshed_img = cv2.threshold(resultBGR,
                                      127, 255, cv2.THRESH_BINARY)
    grayImg = cv2.cvtColor(threshed_img, cv2.COLOR_BGR2GRAY)
    image, contours, hier = cv2.findContours(grayImg, cv2.RETR_TREE,
                                             cv2.CHAIN_APPROX_SIMPLE)

    maxW =0
    maxH =0
    boundX=0
    boundY=0


    for c in contours:
        # get the bounding rect
        x, y, w, h = cv2.boundingRect(c)
        # draw a green rectangle to visualize the bounding rect
        cv2.rectangle(bright, (x, y), (x + w, y + h), (125, 125, 125), 2)

        if (maxW <= w and maxH <= h):
            maxW = w
            maxH = h
            boundX = x
            boundY = y

    boundary = 5
    imgCropped = bright2[boundY - boundary:boundY + maxW + boundary, boundX - boundary:boundX + maxW + boundary]
    cv2.imwrite("images/processed.jpg", imgCropped)




@app.route('/upload_avatar',methods=['POST'])
def upload():
    target = os.path.join(APP_ROOT, 'images/')
    print(target)
    if not os.path.isdir(target):
        os.mkdir(target)

    file = request.files.get('file')
    destination = "/".join([target,"getClass.jpg"])
    file.save(destination)

    processImage(file)

    return jsonify(
        imgClass = "Class 1"
    )

if __name__ == "__main__":
    app.run()


def loadClassifier(classifierName):
    print("loading classifier" + classifierName)

