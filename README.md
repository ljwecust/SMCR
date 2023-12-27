### folders and files:
`/dataset_WIKIMEL` contains the pre-processed input datasets, which could be downloaded from https://github.com/seukgcode/MELBench and use `/main/get_mention_img_info.py` to get different types of descriptions

`/dataset_wikidiverse` contains the pre-processed input datasets, which could be downloaded from https://github.com/wangxw5/wikidiverse and use `/main/get_mention_img_info.py` to get different types of descriptions

`/dataset_WIKIMEL/result` and `/dataset_wikidiverse/result` contain the output results

The processed test set also stored in: https://drive.google.com/drive/folders/1TN_-nUqfv8V9nIPVT1O_vgoE4pNEnkgh?usp=sharing

put `WikiMEL_testset.json` and `WikiMEL_testset_label.json` into `/dataset_WIKIMEL`

put `WikiDiverse_testset.json` and `WikiDiverse_testset_label.json` into `/dataset_wikidiverse`

put `WikiMEL_testset_score.json` and `WikiDiverse_testset_score.json` into `/dataset_wikidiverse/visual_expert/output`

### evaluate:
First, configure the settings in the `/main/params.py` file, such as entering the OpenAI key and specifying the output file name. 

Then, run the `/main/run_main.py` file to make predictions.
