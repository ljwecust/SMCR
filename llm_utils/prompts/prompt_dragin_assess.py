import sys
sys.path.append('..')
import main.params as params
from llm_utils.utils import *

SYSTEM_INFO_INIT = '''You are a natural language processing expert.
I want you to perform entity linking task, linking ambiguous mention in text to its corresponding entity in knowledge base.
I will provide a mention with its context text, as well as {num_cands} candidate entities with their descriptions. Based on all the available information, you need choose one entity that is most likely to correlate to the mention.'''
SYSTEM_INFO = SYSTEM_INFO_INIT.format(num_cands=params.__num_cands_str)


USER_CONTENT_1 = 'GIVEN:\n{ask_dict_1}\nOUTPUT:\nThink step by step and provide the final answer.'


USER_CONTENT_2 = """The simplest way to verify if the selected entity is correct is to replace the mention in the original mention context text with that entity.
If the selected entity is correct, the replaced sentence should remain logical and coherent, and maintain semantic consistency with the original sentence.

Here is a demo:
GIVEN:
`Original`: The Venezuelan delegation at the Maccabiah Games.
`After Replacement`: The Venezuelans delegation at the Maccabiah Games.
OUTPUT:
Based on the given information, the original sentence is..., and the replaced sentence is..., let's analyze carefully to determine if the replaced sentence is reasonable, coherent, and semantically consistent with the original sentence:
(Analysis process)...
<|ASSESSMENT|>: Unreasonable .

Here is another demo:
GIVEN:
`Original`: Moneygall, the village which Obama's great grandfather reportedly comes from.
`After Replacement`: Moneygall, the village which Barack Obama great grandfather reportedly comes from.
OUTPUT:
Based on the given information, the original sentence is..., and the replaced sentence is..., let's analyze carefully to determine if the replaced sentence is reasonable, coherent, and semantically consistent with the original sentence:
(Analysis process)...
<|ASSESSMENT|>: Reasonable .

Now let’s start verifying your answer.
The entity you selected is "{entity1}" and its corresponding description is "{entity1_desc}".
`Original`: {mention_context_text_Original}
`After Replacement`: {mention_context_text_Replaced}
OUTPUT:
Your output must include this key points: <|ASSESSMENT|> .
When providing the final answer, use this format: `<|ASSESSMENT|>: <Reasonable/Unreasonable>`. Please choose either "Reasonable" or "Unreasonable" as your evaluation.
Please compare the mention context text Original and After Replacement, and assess whether the replaced sentence is reasonable, coherent, and semantically consistent with the original one.
The replaced sentence may have some grammatical errors, which is inevitable. Please focus primarily on whether the semantics of the sentence has changed."""



def getPrompt(sample, GPTres, ask_type=None, ve_type=None):

    mention_name, mention_context, mention_imgpath, mention_imgdesc, Cands = getSample(sample)

    if ask_type == 'PTres':
        ask_dict_backbone = {'mention': mention_name,
                             'mention context text': mention_context,
                             'candidate entities': Cands}
        SystemInfo = SYSTEM_INFO

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
        SystemInfo = SYSTEM_INFO.replace('I will provide a mention with its context text,', 'I will provide a mention with its context text and image information,')
    
    candName_list = [i['name'] for i in sample['Candentity']]
    ans = getGPTans(GPTres, candName_list)

    if ans in ['nil', '0', ]:
        return 'break', 'break', 'break', 'break'

    ans_idx = candName_list.index(ans)
    ans_desc = getAheadSentence(sample['Candentity'][ans_idx]['desc'], restrict=224)
    ans = ans.replace('_', ' ')

    mention_context_replaced = mention_context.replace(mention_name, ans)

    user_content_1_new = USER_CONTENT_1.format(ask_dict_1=ask_dict_backbone)
    user_content_2_new = USER_CONTENT_2.format(entity1=ans, entity1_desc=ans_desc, mention_context_text_Original=mention_context, mention_context_text_Replaced=mention_context_replaced)

    return SystemInfo, user_content_1_new, GPTres, user_content_2_new


def askGhatgpt_info_draginAssess(sample, GPTres, ask_type=None, ve_type=None):
    SystemInfo, user_content_1, GPTres_1, user_content_2 = getPrompt(sample, GPTres, ask_type=ask_type, ve_type=ve_type)
    return SystemInfo, user_content_1, GPTres_1, user_content_2


if __name__ == '__main__':
    pass
