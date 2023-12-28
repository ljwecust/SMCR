import json
import os
import copy
from tqdm import tqdm

import sys
sys.path.append('.')
from llm_utils.prompts.prompt_PT_v0 import getPrompt as getPrompt_backbone_pt
from llm_utils.prompts.prompt_WI_v0 import getPrompt as getPrompt_backbone_i
from llm_utils.prompts.prompt_analycom import getPrompt as getPrompt_annlycom
from llm_utils.prompts.prompt_dragin_assess import askGhatgpt_info_draginAssess
from llm_utils.prompts.prompt_dragin_rechoice import askGhatgpt_info_draginRechoice
from llm_utils.utils import *
import params


_status_code_backbone_ask = 0
_status_code_annlycom_ask = 0
_status_code_draginAssess_ask = 0
_status_code_draginRechoice_ask = 0


def loaddata(path_json):
    with open(path_json, 'r', encoding='utf-8') as f: return json.load(f)

def dumpdata(obj, path_json):
    with open(path_json, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=4)
    return

############################## backbone ASK ##############################
def askGhatgpt_info_backbone(sample, ask_type=None, ve_type=None):

    mention_name, mention_context, mention_imgpath, mention_imgdesc, Cands = getSample(sample)

    if ask_type == 'PTres':
        ask_dict = {'mention': mention_name,
                    'mention context text': mention_context,
                    'candidate entities': Cands}
        SystemInfo, user_content = getPrompt_backbone_pt(ask_dict)
    
    elif ask_type == 'Ires':
        if isinstance(mention_imgdesc, dict):
            if isinstance(ve_type, str):
                mention_imgdesc_new = mention_imgdesc[ve_type]
            elif isinstance(ve_type, list):
                mention_imgdesc_new = dict([(key, mention_imgdesc[key]) for key in ve_type])
            else:
                raise
        else:
            mention_imgdesc_new = mention_imgdesc

        ask_dict = {'mention': mention_name,
                    'mention context text': mention_context,
                    'mention image information':mention_imgdesc_new,
                    'candidate entities': Cands}
        SystemInfo, user_content = getPrompt_backbone_i(ask_dict)
    else:
        raise

    return SystemInfo, user_content

def execute_sample_backbone(sample, ask_type=None, ve_type=None, ask_gpt=None, record=False):
    SystemInfo, user_content = askGhatgpt_info_backbone(sample, ask_type=ask_type, ve_type=ve_type)
    messages = [
        {'role':'system', 'content': SystemInfo},
        {'role':'user', 'content': user_content}
    ]

    GPTres = ask_gpt.askGPT4Use_nround(messages)

    global _status_code_backbone_ask
    if record and _status_code_backbone_ask == 0:
        savelog(messages, 'backbone ASK', GPTres)
        _status_code_backbone_ask = 1
    
    return GPTres

############################## annlycom ASK ##############################
def askGhatgpt_info_annlycom(sample, backboneRes_key=''):
    backbone_ans = getBackboneDraginAns(sample, res_key=backboneRes_key, redirect=False, th=40, cut_th=600)
    if backbone_ans in ['0', 'nil']:
        return 'break', 'break', 'break', 'break'

    candName_list = [i['name'] for i in sample['Candentity']]; assert backbone_ans in candName_list
    backbone_ans_idx = candName_list.index(backbone_ans)
    backbone_ans_desc = sample['Candentity'][backbone_ans_idx]['desc']

    ask_dict_annlycom = {'name': '{}. {}'.format(backbone_ans_idx + 1, backbone_ans),
                         'desc': backbone_ans_desc}
    
    mention_name, mention_context, mention_imgpath, mention_imgdesc, Cands = getSample(sample)

    if backboneRes_key == 'PTres':
        ask_dict_backbone = {'mention': mention_name,
                             'mention context text': mention_context,
                             'candidate entities': Cands}
    elif backboneRes_key == 'Ires':
        if isinstance(mention_imgdesc, dict):
            if len(params.__using_mention_imgdesc) == 1:
                mention_imgdesc_new = mention_imgdesc[params.__using_mention_imgdesc[0]]
            else:
                mention_imgdesc_new = dict([(key, mention_imgdesc[key]) for key in params.__using_mention_imgdesc])
        else:
            mention_imgdesc_new = mention_imgdesc
        ask_dict_backbone = {'mention': mention_name,
                             'mention context text': mention_context,
                             'mention image information':mention_imgdesc_new,
                             'candidate entities': Cands}
    else: raise

    GPTres_backbone = sample[backboneRes_key]['backbone']
    
    SystemInfo, user_content_1, GPTres_1, user_content_2 = getPrompt_annlycom(ask_dict_backbone, GPTres_backbone, ask_dict_annlycom)
    return SystemInfo, user_content_1, GPTres_1, user_content_2

def execute_sample_annlycom(sample, ask_gpt):
    SystemInfo, user_content_1, GPTres_1, user_content_2 = askGhatgpt_info_annlycom(sample)
    if SystemInfo == 'break':
        return 'break'
    messages = [
        {'role':'system', 'content': SystemInfo},
        {'role':'user', 'content': user_content_1},
        {'role':'assistant', 'content': GPTres_1},
        {'role':'user', 'content': user_content_2},
    ]

    GPTres_annlycom = ask_gpt.askGPT4Use_nround(messages)
    return GPTres_annlycom

############################## Assess & Rechoice ##############################

def execute_sample_draginAssess_meta(sample, GPTres, ask_type=None, ve_type=None, ask_gpt=None, record=False):
    if GPTres == 'break':
        return 'break'
    
    SystemInfo, user_content_1, GPTres_1, user_content_2 = askGhatgpt_info_draginAssess(sample, GPTres, ask_type=ask_type, ve_type=ve_type)

    if SystemInfo == 'break':
        return 'break'
    
    messages = [
        {'role':'system', 'content': SystemInfo},
        {'role':'user', 'content': user_content_1},
        {'role':'assistant', 'content': GPTres_1},
        {'role':'user', 'content': user_content_2},
    ]
    
    GPTres = ask_gpt.askGPT4Use_nround(messages)

    global _status_code_draginAssess_ask
    if record and _status_code_draginAssess_ask == 0:
        savelog(messages, 'draginAssess ASK', GPTres)
        _status_code_draginAssess_ask = 1

    return GPTres


def execute_sample_draginRechoice_meta(sample, GPTres, GPTassess, ask_type=None, ve_type=None, ask_gpt=None, record=False):
    if GPTassess == 'break':
        return 'break'
    
    SystemInfo, user_content_1, GPTres_1, user_content_2, GPTres_2, user_content_3 = askGhatgpt_info_draginRechoice(sample, GPTres, GPTassess, ask_type=ask_type, ve_type=ve_type)

    if SystemInfo == 'break':
        return 'break'
    
    messages = [
        {'role':'system', 'content': SystemInfo},
        {'role':'user', 'content': user_content_1},
        {'role':'assistant', 'content': GPTres_1},
        {'role':'user', 'content': user_content_2},
        {'role':'assistant', 'content': GPTres_2},
        {'role':'user', 'content': user_content_3},
    ]

    GPTres = ask_gpt.askGPT4Use_nround(messages)

    global _status_code_draginRechoice_ask
    if record and _status_code_draginRechoice_ask == 0:
        savelog(messages, 'draginRechoice ASK', GPTres)
        _status_code_draginRechoice_ask = 1
    
    return GPTres


def execute_sample_dragin(sample, cycle_num=3, ask_type=None, ve_type=None, ask_gpt=None, record=False):
    assert cycle_num in [1,2,3,4,5,]

    assess = {}
    GPTres_backbone = sample['PTIres']['backbone']
    backbone_assess = execute_sample_draginAssess_meta(sample, GPTres_backbone, ask_type=ask_type, ve_type=ve_type, ask_gpt=ask_gpt, record=record)

    res_i = GPTres_backbone; assess_i = backbone_assess
    for i in range(cycle_num):
        if getGPTassess(assess_i) == 'Unreasonable':
            res_i = execute_sample_draginRechoice_meta(sample, res_i, assess_i, ask_type=ask_type, ve_type=ve_type, ask_gpt=ask_gpt, record=record)
            assess_i = execute_sample_draginAssess_meta(sample, res_i, ask_type=ask_type, ve_type=ve_type, ask_gpt=ask_gpt, record=record)
        elif getGPTassess(assess_i) == 'Reasonable':
            res_i = 'break'
            assess_i = 'break'
        assess[i] = [res_i, assess_i]
    return backbone_assess, assess

#

def execute_dataset_wkmel(dataset, save_path, tempfile_path, ask_gpt=None, score_dataset=None, record=False):
    annlycom_th = 29

    def ex_meta(sample, ask_type, ve_type, ask_gpt=ask_gpt, record=record):
        handle_sample = copy.deepcopy(sample)
        handle_sample['PTIres'] = {}

        res_backbone = execute_sample_backbone(handle_sample, ask_type=ask_type, ve_type=ve_type, ask_gpt=ask_gpt, record=record)
        handle_sample['PTIres']['backbone'] = res_backbone

        backbone_assess, assess = execute_sample_dragin(handle_sample, cycle_num=params.__cycle_num, ask_type=ask_type, ve_type=ve_type, ask_gpt=ask_gpt, record=record)
        handle_sample['PTIres']['backbone_assess'] = backbone_assess
        handle_sample['PTIres']['assess'] = assess

        return handle_sample


    for idx, (sample_id, sample) in tqdm(enumerate(dataset.items())):

        if 'GPTans' in sample and sample['GPTans'] not in ['None', '']:
            continue
        
        sample_pt = ex_meta(sample, 'PTres', None)
        ans_0 = getBackboneDraginAns(sample_pt, 'PTIres', False)
        score_0 = get_ans_score(sample_id, sample_pt, score_dataset=score_dataset)
        if score_0 != 'break' and score_0 >= annlycom_th:
            ans = ans_0
        else:
            sample_i_1 = ex_meta(sample, 'Ires', 'OCR text')
            ans_1 = getBackboneDraginAns(sample_i_1, 'PTIres', False)
            score_1 = get_ans_score(sample_id, sample_i_1, score_dataset=score_dataset)
            if score_1 != 'break' and score_1 >= annlycom_th:
                ans = ans_1
            else:
                sample_i_2 = ex_meta(sample, 'Ires', 'Caption')
                ans_2 = getBackboneDraginAns(sample_i_2, 'PTIres', False)
                score_2 = get_ans_score(sample_id, sample_i_2, score_dataset=score_dataset)
                if score_2 != 'break' and score_2 >= annlycom_th:
                    ans = ans_2
                else:
                    sample_i_3 = ex_meta(sample, 'Ires', 'Dense Captions')
                    ans_3 = getBackboneDraginAns(sample_i_3, 'PTIres', False)
                    score_3 = get_ans_score(sample_id, sample_i_3, score_dataset=score_dataset)
                    if score_3 != 'break' and score_3 >= annlycom_th:
                        ans = ans_3
                    else:
                        sample_i_4 = ex_meta(sample, 'Ires', 'Tags')
                        ans_4 = getBackboneDraginAns(sample_i_4, 'PTIres', False)
                        score_4 = get_ans_score(sample_id, sample_i_4, score_dataset=score_dataset)
                        if score_4 != 'break' and score_4 >= annlycom_th:
                            ans = ans_4
                        else:
                            ans = ans_0

        dataset[sample_id]['GPTans'] = ans

        if idx % 10 == 0:
            dumpdata(dataset, tempfile_path)
    dumpdata(dataset, save_path)
    return


def execute_dataset_wkpd(dataset, save_path, tempfile_path, ask_gpt=None, score_dataset=None, record=False):
    annlycom_th = 29

    def ex_meta(sample, ask_type, ve_type, ask_gpt=ask_gpt, record=record):
        handle_sample = copy.deepcopy(sample)
        handle_sample['PTIres'] = {}

        res_backbone = execute_sample_backbone(handle_sample, ask_type=ask_type, ve_type=ve_type, ask_gpt=ask_gpt, record=record)
        handle_sample['PTIres']['backbone'] = res_backbone

        backbone_assess, assess = execute_sample_dragin(handle_sample, cycle_num=params.__cycle_num, ask_type=ask_type, ve_type=ve_type, ask_gpt=ask_gpt, record=record)
        handle_sample['PTIres']['backbone_assess'] = backbone_assess
        handle_sample['PTIres']['assess'] = assess

        return handle_sample


    for idx, (sample_id, sample) in tqdm(enumerate(dataset.items())):

        if 'GPTans' in sample and sample['GPTans'] not in ['None', '']:
            continue
        
        sample_pt = ex_meta(sample, 'PTres', None)
        ans_0 = getBackboneDraginAns(sample_pt, 'PTIres')
        score_0 = get_ans_score(sample_id, sample_pt, score_dataset=score_dataset)
        if score_0 != 'break':
            if score_0 >= annlycom_th:
                ans = ans_0
            else:
                sample_i_1 = ex_meta(sample, 'Ires', 'OCR text')
                ans_1 = getBackboneDraginAns(sample_i_1, 'PTIres')
                score_1 = get_ans_score(sample_id, sample_i_1, score_dataset=score_dataset)
                if score_1 != 'break':
                    if score_1 >= annlycom_th:
                        ans = ans_1
                    else:
                        sample_i_2 = ex_meta(sample, 'Ires', 'Caption')
                        ans_2 = getBackboneDraginAns(sample_i_2, 'PTIres')
                        score_2 = get_ans_score(sample_id, sample_i_2, score_dataset=score_dataset)
                        if score_2 != 'break':
                            if score_2 > annlycom_th:
                                ans = ans_2
                            else:
                                sample_i_3 = ex_meta(sample, 'Ires', 'Dense Captions')
                                ans_3 = getBackboneDraginAns(sample_i_3, 'PTIres')
                                score_3 = get_ans_score(sample_id, sample_i_3, score_dataset=score_dataset)
                                if score_3 != 'break':
                                    if score_3 > annlycom_th:
                                        ans = ans_3
                                    else:
                                        sample_i_4 = ex_meta(sample, 'Ires', 'Tags')
                                        ans_4 = getBackboneDraginAns(sample_i_4, 'PTIres')
                                        score_4 = get_ans_score(sample_id, sample_i_4, score_dataset=score_dataset)
                                        if score_4 != 'break':
                                            if score_4 > annlycom_th:
                                                ans = ans_4
                                            else:
                                                ans = ans_0
                                        else:
                                            ans = ans_4
                                else:
                                    ans = ans_3
                        else:
                            ans = ans_2
                else:
                    ans = ans_1
        else:
            sample_i_1 = ex_meta(sample, 'Ires', 'OCR text')
            ans_1 = getBackboneDraginAns(sample_i_1, 'PTIres')
            score_1 = get_ans_score(sample_id, sample_i_1, score_dataset=score_dataset)
            if score_1 != 'break':
                if score_1 >= annlycom_th:
                    ans = ans_1
                else:
                    sample_i_2 = ex_meta(sample, 'Ires', 'Caption')
                    ans_2 = getBackboneDraginAns(sample_i_2, 'PTIres')
                    score_2 = get_ans_score(sample_id, sample_i_2, score_dataset=score_dataset)
                    if score_2 != 'break':
                        if score_2 > annlycom_th:
                            ans = ans_2
                        else:
                            sample_i_3 = ex_meta(sample, 'Ires', 'Dense Captions')
                            ans_3 = getBackboneDraginAns(sample_i_3, 'PTIres')
                            score_3 = get_ans_score(sample_id, sample_i_3, score_dataset=score_dataset)
                            if score_3 != 'break':
                                if score_3 > annlycom_th:
                                    ans = ans_3
                                else:
                                    sample_i_4 = ex_meta(sample, 'Ires', 'Tags')
                                    ans_4 = getBackboneDraginAns(sample_i_4, 'PTIres')
                                    score_4 = get_ans_score(sample_id, sample_i_4, score_dataset=score_dataset)
                                    if score_4 != 'break':
                                        if score_4 > annlycom_th:
                                            ans = ans_4
                                        else:
                                            ans = ans_0
                                    else:
                                        ans = ans_0
                            else:
                                ans = ans_0
                    else:
                        ans = ans_0
            else:
                ans = ans_0
        

        dataset[sample_id]['GPTans'] = ans

        if idx % 10 == 0:
            dumpdata(dataset, tempfile_path)
    dumpdata(dataset, save_path)
    return


def execute_dataset(dataset, save_path, tempfile_path, ask_gpt=None, score_dataset=None, record=False):
    if params.__dataset_type == 'wikimel':
        execute_dataset_wkmel(dataset, save_path, tempfile_path, ask_gpt=ask_gpt, score_dataset=score_dataset, record=record)
    elif params.__dataset_type == 'wikidiverse':
        execute_dataset_wkpd(dataset, save_path, tempfile_path, ask_gpt=ask_gpt, score_dataset=score_dataset, record=record)
    else:
        raise
    return


