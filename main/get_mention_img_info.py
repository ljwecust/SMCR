import os
# import torch
import json
import time
# from PIL import Image
from tqdm import tqdm
import sys
sys.path.append('')
from llm_utils.utils import *
from llm_utils.visual_expert.VE_azure import ex_dataset


if __name__ == '__main__':
    dataset = loaddata('')
    save_path = ''
    temp_path = ''

    ex_dataset(dataset, save_path, temp_path)

