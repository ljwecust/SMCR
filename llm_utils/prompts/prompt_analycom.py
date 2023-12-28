import sys
sys.path.append('..')
import main.params as params
from llm_utils.utils import *


SYSTEM_INFO_INIT = '''You are a natural language processing expert.
I want you to perform entity linking task, linking ambiguous mention in text to its corresponding entity in knowledge base. 
I will provide a mention with its context text, as well as {num_cands} candidate entities with their descriptions. Based on all the available information, you need choose one entity that is most likely to correlate to the mention. 
After you provide an answer, please extract key points from the description text of the entity you selected, which can serve as supporting evidence to prove that the mention corresponds to the entity you selected.'''
SYSTEM_INFO = SYSTEM_INFO_INIT.format(num_cands=params.__num_cands_str)


USER_CONTENT_1 = 'GIVEN:\n{ask_dict_1}\nOUTPUT:\nThink step by step and provide the final answer.'

USER_CONTENT_2 = """Sure, the entity you have chosen is \"{entity1}\". Now, I will provide you with a more detailed description text of \"{entity1}\". Please extract key points from it, which can serve as supporting evidence to prove that the mention corresponds to the entity \"{entity1}\".
You need to follow these requirements: 
1. The key points you extract must be from the description text of \"{entity1}\" without adding any additional information that does not exist in the description text. 
2. Starting from these key points, it can be demonstrated that the mention corresponds to the entity \"{entity1}\".
3. Please provide the extracted key points directly without explanation or answering other content.

Here is a demo:
GIVEN:
...
OUTPUT:
Paralympic wheelchair rugby competitor, silver medal at 2008 Beijing Paralympics, gold medals at 2012 London and 2016 Rio Paralympics.

Now back to the current task. 
The more detailed description of \"{entity1}\" is as follows: {entity1_description}. Please extract the key points from it.
I hope your answer can be as concise as possible (Within 80 words limit), using phrases and keywords whenever possible.
Please provide the extracted key points directly without explanation or answering other content.
OUTPUT:
"""

TASK_RESNET = """"""


def getPrompt(ask_dict_1, GPTres, ask_dict_current, backboneRes_key=''):
    # ask_dict = {'name': '7. Wojciech Rychlik',
    #             'desc': '...'}
    if backboneRes_key == 'PTres':
        SystemInfo = SYSTEM_INFO
    elif backboneRes_key == 'Ires':
        SystemInfo = SYSTEM_INFO.replace('I will provide a mention with its context text,', 'I will provide a mention with its context text and image information,')

    user_content_1_new = USER_CONTENT_1.format(ask_dict_1=ask_dict_1)

    entity_name = ask_dict_current['name']
    entity_desc = ask_dict_current['desc']
    user_content_2_new = USER_CONTENT_2.format(entity1=entity_name, entity1_description={entity_name: entity_desc})

    return SystemInfo, user_content_1_new, GPTres, user_content_2_new



if __name__ == '__main__':
    pass