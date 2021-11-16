import os 

'''import torch
TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
!pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/$CUDA_VERSION/torch$TORCH_VERSION/index.html'''


def setup_DeepAnimalToolkit():
    dependencies = ['pip install -r DeepAnimalToolkit/requirements.txt',
                    'pip install pyyaml==5.1',
                    'pip install torch==1.8.0+cu101 torchvision==0.9.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html',
                    'pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.8/index.html']
    
    for d in dependencies: 
        os.system(d)
        
def setup_demo():
    dependencies = ['wget https://www.dropbox.com/s/04cxtviaercx2y1/single_mouse_dataset.zip', 
                    'wget https://www.dropbox.com/s/28oprx3pcbso19i/single_mouse_test_video.avi',
                    'unzip single_mouse_dataset.zip']
    
    for d in dependencies: 
        os.system(d)
    return 0 