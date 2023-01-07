import time
import flask
from flask import Flask, request, Response, send_file, jsonify, make_response
import json
import sys
sys.path.append('/workspace/customized/tools')
from model_inference import Model
import io
import gc
# load model:
model_class = Model(mode='inference')

# resfulAPI:
app = Flask(__name__)

@app.route('/data', methods=['GET', 'POST'])
def server_obo():
    if request.method == 'POST':
        r = request
        data = r.data
        parm_dict = json.loads(data)
        model_class.imgs_json = parm_dict
        # model_class.load_images()
        t1 = time.time()
        img_dict = model_class.get_model_result_for_client()
        print(round(time.time()-t1, 2))
        return jsonify(img_dict)

@app.route('/test', methods=['GET', 'POST'])
def server_():
    if request.method == 'POST':
        r = request
        data = r.data
        parm_dict = json.loads(data)
        model_class.load_images_name_from_client(parm_dict['image_names'])
        # model_class.load_images()
        t1 = time.time()
        img_dict = model_class.get_model_result_for_result(save_json=parm_dict['save_json'], save_img=parm_dict['save_images'])
        print(round(time.time()-t1, 2))
        return jsonify(img_dict)
        # return json.dumps({'success':True}), 200, {'ContentType':'application/json'} 

@app.route('/inference', methods=['GET', 'POST'])
def inference():
    if request.method == 'POST':
        print(11111)
        model_class.imgs_json = request.files
        t1 = time.time()
        output_image = int(request.values.get('output_image'))
        img_dict = model_class.get_model_result_for_client(output_image=output_image)
        print(round(time.time()-t1, 2))
        gc.collect()
        # return jsonify(img_dict)
        if output_image:
            return send_file(
                io.BytesIO(img_dict['images'][0]),
                mimetype='image/jpeg',
                as_attachment=True,
                attachment_filename='1.jpg')
        else:
            return jsonify(img_dict)

@app.route('/inference_pipeline', methods=['POST'])
def inference_pipeline():
    if request.method == 'POST':
        # print(request.headers)
        send_json = request.json
        print(send_json.keys())
        model_class.imgs_json = send_json
        t1 = time.time()
        output_image = int(send_json.pop('output_image'))
        output_result = int(send_json.pop('output_result'))
        print(output_image, output_result)
        img_dict = model_class.get_model_result_for_client(
            output_image=output_image, 
            pipeline=True,
            output_result=output_result
        )
        print(round(time.time()-t1, 2))
        gc.collect()
        # return jsonify(img_dict)
        if output_image:
            return send_file(
                io.BytesIO(img_dict['images'][0]),
                mimetype='image/jpeg',
                as_attachment=True,
                attachment_filename='1.jpg')
        else:
            return jsonify(img_dict)
# app.run(host='0.0.0.0', port=7777, debug=False, threaded=False)
