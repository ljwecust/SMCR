# -*- coding: UTF-8 -*-

import os

"""run which dataset"""
# __dataset_type = 'wikimel'
__dataset_type = 'wikidiverse'


"""num of parallel processes"""
__num_cut = 10


"""openai KEY"""
key = """"""


"""result save name"""
save_path = 'test1.json'




#######################################################
if __dataset_type == 'wikidiverse':
    __use_wikidiverse_bias_resnet = True
elif __dataset_type == 'wikimel':
    __use_wikidiverse_bias_resnet = False

__num_cands = 10
_int2str = {10:'ten', 12:'twelve', 15:'fifteen', }
__num_cands_str = _int2str[__num_cands]

__cycle_num = 3

__wkeml_score_path = './visual_expert/output/WikiMEL_testset_score.json'
__wkpd_score_path = './visual_expert/output/WikiDiverse_testset_score.json'

save_root = './dataset_WIKIMEL/result/' if __dataset_type == 'wikimel' else './dataset_wikidiverse/result/'
__save_path = os.path.join(save_root, save_path)

__keys = '\n'.join([key] * __num_cut)
