import os
import re
import json
import copy
import numpy as np
import Levenshtein
from PIL import Image
from fuzzywuzzy import fuzz
from fuzzywuzzy import process as fuzzywuzzy_process
from nltk import sent_tokenize
from typing import List, Dict


assert Levenshtein.__version__[:4] == '0.21'

def loaddata(path_json):
    with open(path_json, 'r', encoding='utf-8') as f: return json.load(f)

def dumpdata(obj, path_json):
    with open(path_json, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=4)
    return

def getAheadSentence(text, restrict=120, rm_parentheses=False):
    if rm_parentheses: text = re.sub(u"\\(.*?\\)", "", text)
    sentence_list = sent_tokenize(text)
    ahead_sentence = ''
    for i in range(len(sentence_list)):
        ahead_sentence = ahead_sentence + ' ' + sentence_list[i]
        if len(ahead_sentence) >= restrict:
            break
    return ahead_sentence[1:]

def getAheadWords(text, restrict=500):
    text_list = text.split(' ')
    return ' '.join(text_list[:restrict])

def getSample(sample):
    mention_name = sample['mention']
    mention_context = sample['mention_context']
    mention_imgpath = sample['mention_imgpath']
    try:
        mention_imgdesc = sample['mention_imgdesc_Azure']
    except KeyError:
        mention_imgdesc = 'No relevant image was provided.'

    Candentity_list = [[d['name'].replace('_',' '), getAheadSentence(d['desc'])] for d in sample['Candentity']]
    Cands = {'{}. {}'.format(i+1, Candentity_list[i][0]):Candentity_list[i][1] for i in range(len(Candentity_list))}

    return mention_name, mention_context, mention_imgpath, mention_imgdesc, Cands


def getGPTans(GPTres: str, candNameList: List[str]):
    map_dict = {'à': '%C3%A0', 'á': '%C3%A1', 'â': '%C3%A2', 'ã': '%C3%A3', 'ä': '%C3%A4', 'å': '%C3%A5', 'æ': '%C3%A6', 'ç': '%C3%A7', 'è': '%C3%A8', 'é': '%C3%A9', 'ê': '%C3%AA', 'ë': '%C3%AB', 'ì': '%C3%AC', 'í': '%C3%AD', 'î': '%C3%AE', 'ï': '%C3%AF', 'ð': '%C3%B0', 'ñ': '%C3%B1', 'ò': '%C3%B2', 'ó': '%C3%B3', 'ô': '%C3%B4', 'õ': '%C3%B5', 'ö': '%C3%B6', 'ō': '%C5%8D', '÷': '%C3%B7', 'ø': '%C3%B8', 'ù': '%C3%B9', 'ú': '%C3%BA', 'û': '%C3%BB', 'ü': '%C3%BC', 'ý': '%C3%BD', 'þ': '%C3%BE', 'ÿ': '%C3%BF', '&':'%26', '\"':'%22', '\'':'%27'}
    def getAnswer_s(GPTres, candName_list):
        answer = GPTres.split('ANSWER')[-1]
        answer = answer.split('\n')[0].split('. ')[-1].split(': ')[-1]
        if answer.replace('_', ' ') in candName_list or answer.replace(' ', '_') in candName_list:
            return answer.replace(' ', '_')
        else: return 'fail'
    
    def getAnswer_map(GPTres, candName_list):
        answer = GPTres.split('ANSWER')[-1]
        for w in answer:
            if w in map_dict:
                answer = answer.replace(w, map_dict[w])
        pred_label = []
        for entity_name in candName_list:
            if entity_name.replace(' ', '_') in answer or entity_name.replace('_', ' ') in answer:
                pred_label.append(entity_name)
        return pred_label

    def getAnswer_multi(GPTres, candName_list):
        answer_multi = GPTres.split('ANSWER')[1] if len(GPTres.split('ANSWER')) > 2 else GPTres.split('ANSWER')[-1]
        pred_label = []
        for entity_name in candName_list:
            if entity_name.replace(' ', '_') in answer_multi or entity_name.replace('_', ' ') in answer_multi:
                pred_label.append(entity_name)
        return pred_label

    candName_list = copy.deepcopy(candNameList); candName_list.append('nil')
    answer = GPTres.split('ANSWER')[-1]
    pred_label = []
    for entity_name in candName_list:
        if entity_name.replace(' ', '_') in answer or entity_name.replace('_', ' ') in answer:
            pred_label.append(entity_name)
    if len(pred_label) == 0:
        pred_label = getAnswer_map(GPTres, candName_list)
    if len(pred_label) == 0:
        pred_label = getAnswer_multi(GPTres, candName_list)

    if len(pred_label) == 1:
        return pred_label[0]
    elif len(pred_label) == 0:
        answer_fuzz = answer.split('\n')[0].split('. ')[-1].split(': ')[-1]
        answer_fuzz = answer_fuzz.replace(' ', '_')
        redirect_ans = fuzzywuzzy_process.extractOne(answer_fuzz, [i for i in candName_list if i != 'nil'])
        return redirect_ans[0]
        return '0'
    else:
        res = 'fail'
        # res = getAnswer_s(GPTres, candName_list)
        if res != 'fail':
            return res
        else:
            return max(pred_label, key=len, default='nil')

def getBackboneAnswer(sample, res_key=None):
    candName_list = [i['name'] for i in sample['Candentity']]

    GPTres = sample[res_key]['backbone']

    return getGPTans(GPTres, candName_list)


def getMaxIdx(L: List[float]):
    L_max = max(L)
    max_idx = []
    for _ in range(L.count(L_max)):
        idx = L.index(L_max)
        max_idx.append(idx)
        L[idx] = min(L)
    return max_idx

def convert_to_english(word:str):
    convert_dict = {'à':'a', 'â':'a', 'ä':'a', 'è':'e', 'é':'e', 'ê':'e', 'ë':'e', 'ì':'i', 'î':'i', 'ï':'i', 
                    'ò':'o', 'ó':'o', 'ô':'o', 'ö':'o', 'ù':'u', 'û':'u', 'ü':'u', 'ç':'c', 'œ':'o', '€':'e'}
    source_letter = [k for k in convert_dict]
    for w in word:
        if w in source_letter:
            word = word.replace(w, convert_dict[w])
    return word

def getIdentityEntity(ans_entity:str, Candentity:List[Dict], th=50, cut_th=450):
    candname_list = [i['name'] for i in Candentity]
    assert ans_entity in candname_list

    ans_entity_idx = candname_list.index(ans_entity)
    ans_entity_desc = re.sub(u"\\(.*?\\)", "", Candentity[ans_entity_idx]['desc'])
    if cut_th != -1: ans_entity_desc = ans_entity_desc[:cut_th]

    IdentityEntity_list = []
    for cand in Candentity:
        cand_desc = re.sub(u"\\(.*?\\)", "", cand['desc'])
        if cut_th != -1: cand_desc = cand_desc[:cut_th]
        if fuzz.partial_ratio(ans_entity_desc, cand_desc) >= th:
            IdentityEntity_list.append(cand['name'])
    return IdentityEntity_list

def redirectAns(ans:str, Candentity:List[Dict], th=50, cut_th=450):
    if ans in ['0', 'nil']:
        return ans
    
    IdentityEntity_list = getIdentityEntity(ans, Candentity, th=th, cut_th=cut_th)
    IdentityEntity_list.remove(ans)

    ans_en = convert_to_english(ans)
    to_output_list_1 = [i for i in IdentityEntity_list if i in ans_en]
    if len(to_output_list_1) != 0:
        return min(to_output_list_1, key=len)
    
    if ',_' in ans_en:
        ans_en_reverse = '_'.join(ans_en.split(',_')[::-1])
        to_output_list_2 = [i for i in IdentityEntity_list if i in ans_en_reverse]
        if len(to_output_list_2) != 0:
            return min(to_output_list_2, key=len)

    ans_en_list = ans_en.split('_')
    if ans_en_list[-1][0]=='(' and ans_en_list[-1][-1]==')':
        ans_en_p2d = '_'.join(ans_en_list[:-1]) + ',_' + ans_en_list[-1][1:-1]
        if ans_en_p2d in IdentityEntity_list:
            return ans_en_p2d
    
    return ans

def getBackboneAns_redirect(sample, res_key='PTres', th=50, cut_th=450):
    
    ans = getBackboneAnswer(sample, res_key)
    Candentity = sample['Candentity']

    return redirectAns(ans, Candentity, th=th, cut_th=cut_th)


def getGPTassess(GPTassess:str):
    GPTassess = GPTassess.split('ASSESSMENT')[-1]
    if 'Unreasonable' in GPTassess:
        return 'Unreasonable'
    else:
        return 'Reasonable'

def getBackboneDraginAns(sample, res_key=None, redirect=True, th=50, cut_th=450, num_dragin=3):
    assert res_key in ['PTres', 'Ires', 'PTIres']

    candName_list = [i['name'] for i in sample['Candentity']]
    backbone_ans = getGPTans(sample[res_key]['backbone'], candName_list)

    ans = backbone_ans
    if getGPTassess(sample[res_key]['backbone_assess']) == 'Unreasonable':
        for cycle_num, cycle_res in sample[res_key]['assess'].items():
            if int(cycle_num) >= num_dragin:
                break
            if cycle_res[1] == 'break':
                break

            if getGPTassess(cycle_res[1]) == 'Unreasonable':
                continue
            else:
                if cycle_res[0] == 'break':
                    break
                if getGPTans(cycle_res[0], candName_list) not in candName_list:
                    break

                ans = getGPTans(cycle_res[0], candName_list)
    
    if redirect == True:
        ans = redirectAns(ans, sample['Candentity'], th=th, cut_th=cut_th)
    
    return ans


def get_ans_score(sample_id, sample, score_dataset):
    ans = getBackboneDraginAns(sample, 'PTIres', redirect=False)

    Candentity_list = score_dataset[sample_id]['Candentity']
    candName_list = [i['name'] for i in Candentity_list]

    if ans in candName_list:
        ans_idx = candName_list.index(ans)
        ans_score = Candentity_list[ans_idx]['score']
    else:
        ans_score = 'break'
    
    return ans_score


def savelog(messages_list, info_title, GPT_res='', save_path='./log.txt'):
    with open(save_path, 'a', encoding='utf-8') as f:
        f.write('{} {} {}\n'.format('='*64, info_title, '='*64))

        for d in messages_list:
            f.write('@@@{}@@@\n'.format(d['role']))
            if isinstance(d['content'], str):
                f.write(d['content'])
                f.write('\n')
            else:
                for dd in d['content']:
                    f.write(dd.get('text') if dd.get('text') != None else dd.get('image_url')['url'])
                f.write('\n')
        f.write('\n')
        f.write('@@@ GPT_res @@@:\n')
        f.write(GPT_res)
        f.write('\n')
    return


