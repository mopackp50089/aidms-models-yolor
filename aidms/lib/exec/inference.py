import time
from argparse import ArgumentParser
import sys
sys.path.append('/workspace/customized/tools')
from model_inference import Model
# load model:
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--mode", type=str, default='test', help="set mode, ('test' | 'test_uploadimg')")
    args = parser.parse_args()
    model_class = Model(mode=args.mode)
    # model_class.load_images()
    # t1 = time.time()
    save_img = False if args.mode == 'test' else True
    img_dict = model_class.get_model_result_for_result(save_img=save_img)
    # print(round(time.time()-t1, 2))
