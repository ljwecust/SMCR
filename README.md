# SMCR

Download all code and data: [SMCR_all.zip](https://drive.google.com/file/d/1plqHyvXFzU__xWxbORomYAQ2jNfRBOiP/view?usp=sharing)

### folders and files:
`./dataset_WIKIMEL`and `./dataset_wikidiverse` contains the pre-processed input datasets, which could be downloaded from [MELBench](https://github.com/seukgcode/MELBench) and [WikiDiverse](https://github.com/wangxw5/wikidiverse). Use `./main/get_mention_img_info.py` to get different types of descriptions

The processed test set also stored in [here](https://drive.google.com/drive/folders/1TN_-nUqfv8V9nIPVT1O_vgoE4pNEnkgh?usp=sharing)

put `WikiMEL_testset.json` and `WikiMEL_testset_label.json` into `./dataset_WIKIMEL`

put `WikiDiverse_testset.json` and `WikiDiverse_testset_label.json` into `./dataset_wikidiverse`

put `WikiMEL_testset_score.json` and `WikiDiverse_testset_score.json` into `./dataset_wikidiverse/visual_expert/output`

### evaluate:
First, in the `./main/params.py` file, set the openai-key, specify the output file name, and specify the dataset for evaluation.

Second, running the `./main/run_main.py` script for prediction.

The results are saved in `/dataset_WIKIMEL/result` and `/dataset_wikidiverse/result`
