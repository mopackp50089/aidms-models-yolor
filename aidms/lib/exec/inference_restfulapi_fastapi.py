import time

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel
from starlette.responses import StreamingResponse
import uvicorn

import json
import sys
sys.path.append('/workspace/customized/tools')
from model_inference import Model
import io
import gc

import pickle


# load model:
model_class = Model(mode='inference')

# resfulAPI:

class ServerObo(BaseModel):
    data : str

app = FastAPI(debug=False)

@app.post("/data/")
def server_obo(obo : ServerObo):
    parm_dict = json.loads(obo.data)
    model_class.imgs_json = parm_dict
    # model_class.load_images()
    t1 = time.time()
    img_dict = model_class.get_model_result_for_client()
    print(round(time.time()-t1, 2))
    return jsonify(img_dict)

@app.post("/test/")
def server_(obo : ServerObo):
    parm_dict = json.loads(obo.data)
    model_class.load_images_name_from_client(parm_dict['image_names'])
    # model_class.load_images()
    t1 = time.time()
    img_dict = model_class.get_model_result_for_result(save_json=parm_dict['save_json'], save_img=parm_dict['save_images'])
    print(round(time.time()-t1, 2))
    return jsonify(img_dict)
    # return json.dumps({'success':True}), 200, {'ContentType':'application/json'} 

@app.post("/inference/{output_image}")
async def inference(output_image : int, image: UploadFile = File(...)):
    contents = await image.read()
    model_class.imgs_json = [{'image': contents,'filename': image.filename}]
    output_image = int(output_image)
    img_dict = model_class.get_model_result_for_client(output_image=output_image)
    gc.collect()
    if output_image:
        return Response(content = img_dict['images'][0], headers = {}, media_type="image/png")
    else:
        for index in range(len(img_dict['classes'][0])):
            img_dict['classes'][0][index] = classlist[img_dict['classes'][0][index]]
        return json.dumps(img_dict)

if __name__=='__main__':
    uvicorn.run(app='inference_restfulapi_fastapi:app',host='0.0.0.0',port=7777,reload=True)
