import os
import tempfile
import time
import argparse

parser = argparse.ArgumentParser(description="Description of your program")

parser.add_argument(
"-m", "--model_path", help="path to a model - this will be a .pth file", required=True
) 

parser.add_argument(
"-t", "--model_type", help="what kind of model is being used", required=True
)

parser.add_argument(
"-p", "--partition", help="name of an accesable slurm partition", required=True
)

parser.add_argument(
"-a", "--account", help="name of a slurm account", required=True
)

parser.add_argument(
"-v", "--videos", help="path to videos folder", required=True
)
args = vars(parser.parse_args())

videos = [x for x in os.listdir(args['videos']) if x[-4:] == ".avi"]

for video in videos:
    output = video.split('.')[0]+'_tracking'
    
    SBATCH_STRING = """#!/bin/sh
    
#SBATCH --account={account}
#SBATCH --partition={partition}
#SBATCH --job-name={jobname}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=40GB

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:{fill in your path to conda lib folder}

export PATH={fill in your path to conda bin folder}:$PATH

cd {fill in your path to this directory path/ending/in/DAT}

python track_single_instance.py -v ' + {video} + ' -o ' + {output} + ' -m ' + {model} + ' -t ' + {model_type}

"""
    

    SBATCH_STRING = SBATCH_STRING.format(
        video = video,
        jobname = 'deepAnimalToolkit',
        output = output,
        model = args["model_path"],        
        model_type = args["model_type"],
        partition = args["partition"],
        account = args["account"]
    )
    
    dirpath = tempfile.mkdtemp()

    with open(os.path.join(dirpath, "scr.sh"), "w") as tmpfile:
        tmpfile.write(SBATCH_STRING)
    os.system(f"sbatch {os.path.join(dirpath, 'scr.sh')}")
    print(f"Launched from {os.path.join(dirpath, 'scr.sh')}")
    time.sleep(1)
