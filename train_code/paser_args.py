import argparse
import os
def createDirectory(*directory):
    for i in directory:
        try:
            if not os.path.exists(i):
                os.makedirs(i)
        except OSError:
            print("Error: Failed to create the directory.")

def parse_args(name = None):
    path = os.getcwd()
    
    classes_temp = classes_temp = ["Cataract","Corneal_Ulcer","Eyelid_Tumor","Normal","Uveitis"]
    parser = argparse.ArgumentParser('tbi')
    parser.add_argument('--save_path', type=str, default=os.path.join(path,'result'))
    parser.add_argument('--save_path2', type=str, default=os.path.join(path,'result2'))
    parser.add_argument('--save_path_best', type=str, default=os.path.join(path,'best_one_gpu'))
    parser.add_argument('--save_path_best2', type=str, default=os.path.join(path,'best_two_gpu'))
    parser.add_argument('--save_path_report', type=str, default=os.path.join(path,'report'))
    parser.add_argument('--save_path_conpusion', type=str, default=os.path.join(path,'conpusion'))
    parser.add_argument('--arch', type=str, default=name)
    parser.add_argument('--classes',type=list, default = classes_temp)
    createDirectory('result','result2','best_one_gpu','best_two_gpu','report','conpusion')
    return parser.parse_args()