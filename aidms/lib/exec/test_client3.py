import base64
import requests
import cv2
import numpy as np
import json
from pathlib import Path
import time
from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--image_path", type=str, help="image path")
    args = parser.parse_args()
    # while True:
    while True:
        send_data = {'image': open(args.image_path, 'rb')}
        
        # return image or not:
        output_image = 0
        data = {'output_image': output_image}

        url = 'http://0.0.0.0:7777/inference'
        print('Send image ...')
        t1 = time.time()
        response = requests.post(url, data=data, files=send_data)
        print(time.time()-t1)


    if not output_image:
        print(response.json())
    else:
        b_str = response.content
        nparr = np.fromstring(b_str, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        print(img.shape)
        cv2.imwrite('test.jpg', img)



