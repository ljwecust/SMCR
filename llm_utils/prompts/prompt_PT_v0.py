import sys
sys.path.append('..')
import main.params as params


SYSTEM_INFO_INIT = '''You are a natural language processing expert.
I want you to perform entity linking task, linking ambiguous mention in text to its corresponding entity in knowledge base.
I will provide a mention with its context text, as well as {num_cands} candidate entities with their descriptions. Based on all the available information, you need choose one entity that is most likely to correlate to the mention.'''
SYSTEM_INFO = SYSTEM_INFO_INIT.format(num_cands=params.__num_cands_str)


SYSTEM_INFO_ADD_1 = '''
If none of the candidate entities correspond to the mention, you ANSWER: nil.'''
# you need choose one entity to answer what does the MENTION refer to.
# you need choose one entity that is most likely to correlate to the mention.


TASK_COT_ICL = """GIVEN:
{'mention': 'Governor Ed Rendell', 'mention context text': 'Governor Ed Rendell announced…', 'candidate entities': {'1. Don Rendell': '…', '2. Stephen Rendell March': '…', '3. Ed Rendell': '…', '4. Stuart Rendell': '…', '5. Midge Rendell': '…', '6. Kenneth W. Rendell': '…', '7. Edward "Ed" G. Rendell': '…', '8. Ruth Rendell': '…', '9. Mike Rendell': '…', '10. Rendell': '…'}}
OUTPUT:
Based on the given information, the mention is "Governor Ed Rendell" and the mention context text is… We need to select one entity from the ten candidate entities that is most likely to correlate with the mention. Let's go through the steps to determine the correct entity:
Step 1: Analyze the mention and context:
...

Step 2: Compare the mention with the candidate entities:
Let's go through each candidate entity with its description and determine if they have any association with the mention:
(In this step, compare the mention to the entity in as much detail as possible)
...

Step 3: Select the most relevant candidate entities:
...

<|ANSWER|>: 3. Ed Rendell
"""
# Based on the above analysis, determine what "Governor Ed Rendell" refers to in "Governor Ed Rendell announced…":
# Select the most relevant candidate entities:
# Based on the above analysis, determine what "{MENTION}" refers to in the text "{mention context text}"


TASK_RESNET = """Your output must include this key points: <|ANSWER|> .
When providing the final answer, use this format: `<|ANSWER|>: <your answer>`. <your answer> must be only one option from {CAND_LIST}; do not provide multiple candidate entities and avoid irrelevant responses."""

# TASK_RESNET = """Your output must include this key points: <|ANSWER|> .
# When providing the final answer, use this format: `<|ANSWER|>: <your answer>`. <your answer> must be only one option from {CAND_LIST}"""

TASK_RESNET_ADD_1 = """
If none of the candidate entities correspond to the mention, you ANSWER: nil."""

TASK_RESNET_ADD_2 = """
Among multiple candidate entities, the one with the simplest name is more likely to be the answer. For example, in [1. Wikipedia (website), 2. German Wikipedia, 3. Wikipedia], the answer is 3. Wikipedia. Of course, this rule doesn't always work."""

# Among multiple candidate entities, the one with a name most similar to the mention is more likely to be the answer. For example, if the mention is "Canadian" and the candidate entity list is [1. The Canadian (film), 2. Canada Sings, 3. Canada], the answer is 3. Canada.


def getPrompt(ask_dict, dataset_type=params.__dataset_type, use_wikidiverse_bias=params.__use_wikidiverse_bias_resnet):
    assert dataset_type in ['wikidiverse', 'wikimel']

    cand_list = [k for k,v in ask_dict['candidate entities'].items()]

    if dataset_type == 'wikidiverse':
        cand_list.append('nil')
        SystemInfo = SYSTEM_INFO + SYSTEM_INFO_ADD_1
        task_resnet_temp_1 = TASK_RESNET + TASK_RESNET_ADD_1
    elif dataset_type == 'wikimel':
        SystemInfo = SYSTEM_INFO
        task_resnet_temp_1 = TASK_RESNET
    else:
        raise

    if use_wikidiverse_bias == True:
        task_resnet = task_resnet_temp_1 + TASK_RESNET_ADD_2
    else:
        task_resnet = task_resnet_temp_1

    task_resnet_new = task_resnet.format(CAND_LIST=cand_list)

    system_content = SystemInfo
    user_content = f'Here is a demo:\n{TASK_COT_ICL}\nnow GIVEN:\n{ask_dict}\nOUTPUT:\n{task_resnet_new}'
    return system_content, user_content

if __name__ == '__main__':
    pass
