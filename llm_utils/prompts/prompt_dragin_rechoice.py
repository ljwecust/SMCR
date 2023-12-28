import sys
sys.path.append('..')
import main.params as params
from llm_utils.utils import *


SYSTEM_INFO_INIT = '''You are a natural language processing expert.
I want you to perform entity linking task, linking ambiguous mention in text to its corresponding entity in knowledge base.
I will provide a mention with its context text, as well as {num_cands} candidate entities with their descriptions. Based on all the available information, you need choose one entity that is most likely to correlate to the mention.'''
SYSTEM_INFO = SYSTEM_INFO_INIT.format(num_cands=params.__num_cands_str)


SYSTEM_INFO_ADD = """
If none of the candidate entities correspond to the mention, you ANSWER: nil."""


USER_CONTENT_1 = 'GIVEN:\n{ask_dict_1}\nOUTPUT:\nThink step by step and provide the final answer.'


USER_CONTENT_2 = """The simplest way to verify if the selected entity is correct is to replace the mention in the original mention context text with that entity.
If the selected entity is correct, the replaced sentence should remain logical and coherent, and maintain semantic consistency with the original sentence.

Now let’s start verifying your answer.
The entity you selected is "{entity1}" and its corresponding description is "{entity1_desc}".
`Original`: {mention_context_text_Original}
`After Replacement`: {mention_context_text_Replaced}
OUTPUT:
Tell me if the sentence after replacement is Reasonable. If it is Unreasonable, then your selected entity might be wrong."""


USER_CONTENT_3 = """Okay, since you think the sentence after the replacement is Unreasonable, it means that the entity you previously chose may have been incorrect. Now please reconsider and provide the entity corresponding to the mention "{mention}".
Your output must include this key points: <|ANSWER|> .
When providing the final answer, use this format: `<|ANSWER|>: <your answer>`. <your answer> must be only one option from {CAND_LIST}; do not provide multiple candidate entities and avoid irrelevant responses."""


USER_CONTENT_3_ADD_1 = """
If none of the candidate entities correspond to the mention, you ANSWER: nil."""


USER_CONTENT_3_ADD_2 = """
Among multiple candidate entities, the one with the simplest name is more likely to be the answer. For example, in [1. Wikipedia (website), 2. German Wikipedia, 3. Wikipedia], the answer is 3. Wikipedia. Of course, this rule doesn't always work."""



def getPrompt(sample, GPTres, GPTassess, ask_type=None, ve_type=None, dataset_type=params.__dataset_type):

    mention_name, mention_context, mention_imgpath, mention_imgdesc, Cands = getSample(sample)

    cand_list = [k for k,v in Cands.items()]
    
    if dataset_type == 'wikidiverse':
        cand_list.append('nil')
        SystemInfo_temp_1 = SYSTEM_INFO + SYSTEM_INFO_ADD
        USER_CONTENT_3_temp_1 = USER_CONTENT_3 + USER_CONTENT_3_ADD_1
    elif dataset_type == 'wikimel':
        SystemInfo_temp_1 = SYSTEM_INFO
        USER_CONTENT_3_temp_1 = USER_CONTENT_3
    else: raise


    if ask_type == 'PTres':
        ask_dict_backbone = {'mention': mention_name,
                             'mention context text': mention_context,
                             'candidate entities': Cands}
        SystemInfo = SystemInfo_temp_1
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

        ask_dict_backbone = {'mention': mention_name,
                             'mention context text': mention_context,
                             'mention image information':mention_imgdesc_new,
                             'candidate entities': Cands}
        SystemInfo = SystemInfo_temp_1.replace('I will provide a mention with its context text,', 'I will provide a mention with its context text and image information,')
    

    candName_list = [i['name'] for i in sample['Candentity']]
    ans = getGPTans(GPTres, candName_list)
    if ans in ['nil', '0', ]:
        return 'break', 'break', 'break', 'break', 'break', 'break'

    ans_idx = candName_list.index(ans)
    ans_desc = getAheadSentence(sample['Candentity'][ans_idx]['desc'], restrict=224)
    ans = ans.replace('_', ' ')

    mention_context_replaced = mention_context.replace(mention_name, ans)

    user_content_1_new = USER_CONTENT_1.format(ask_dict_1=ask_dict_backbone)
    user_content_2_new = USER_CONTENT_2.format(entity1=ans, entity1_desc=ans_desc, mention_context_text_Original=mention_context, mention_context_text_Replaced=mention_context_replaced)

    if params.__use_wikidiverse_bias_resnet == True:
        user_content_3_temp_2 = USER_CONTENT_3_temp_1 + USER_CONTENT_3_ADD_2
    else:
        user_content_3_temp_2 = USER_CONTENT_3_temp_1
    
    user_content_3_new = user_content_3_temp_2.format(mention=mention_name, CAND_LIST=cand_list)
    return SystemInfo, user_content_1_new, GPTres, user_content_2_new, GPTassess, user_content_3_new



def askGhatgpt_info_draginRechoice(sample, GPTres, GPTassess, ask_type=None, ve_type=None, dataset_type=params.__dataset_type):
    SystemInfo, user_content_1, GPTres_1, user_content_2, GPTres_2, user_content_3 = getPrompt(sample, GPTres, GPTassess, ask_type=ask_type, ve_type=ve_type, dataset_type=dataset_type)
    return SystemInfo, user_content_1, GPTres_1, user_content_2, GPTres_2, user_content_3


if __name__ == '__main__':
    pass
