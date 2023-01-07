import os
import sys
sys.path.append('/workspace/')
sys.path.append('/workspace/customized/models/')
from preprocessing import bash_command
from model_parameters import Model_Parameters
model_parameters = Model_Parameters(step='training')

def main():
    return_code, stdout, stderr = bash_command(f"python /workspace/customized/models/preprocessing.py")
    if return_code:
        print(stderr)
        sys.exit(1)
    print('Preprocessing finish')

    #step1.change working directory 
    os.chdir(f"{model_parameters.model_location}")
    
    #step2.add your model training command 
    #Ex: return_code, stdout, stderr = bash_command(f" your model training command ")
    return_code, stdout, stderr = bash_command(f"python train.py")
    if return_code:
        print(stderr)
        sys.exit(1)
    print('Model has trained.')
    
if __name__ == '__main__':
    main()

