# Fine-Tuning Pre-Trained Language Models with Gaze Supervision

In this paper, we propose to integrate a gaze module into pre-trained language models at the fine-tuning stage to improve their capabilities to learn representations that are grounded in human language processing. This is done by extending the conventional purely text-based fine-tuning objective with an auxiliary loss to exploit cognitive signals. The gaze module is only included during training, retaining compatibility with existing PLM-based pipelines. We evaluate the proposed approach using two distinct PLMs on the GLUE benchmark and observe that the proposed model improves performance compared to both standard fine-tuning and traditional text augmentation baselines.

## Setup
Clone repository:

```
git clone https://github.com/aeye-lab/ACL-GazeSupervisedLM
```

Install dependencies:

```
pip install -r requirements.txt
```

Download precomputed sn_word_len mean and std (from CELER dataset) for Eyettention model feature normalization:

```
wget https://github.com/aeye-lab/ACL-GazeSupervisedLM/releases/download/v1.0/bert_feature_norm_celer.pickle
wget https://github.com/aeye-lab/ACL-GazeSupervisedLM/releases/download/v1.0/roberta_feature_norm_celer.pickle
```

## Run Experiments
To reproduce the results in Section 3.2:
```
bash run_glue_gazesup_bert_low_resource.sh
bash run_glue_gazesup_roberta_low_resource.sh
```

To reproduce the results in Section 3.3:
```
bash run_glue_gazesup_bert_high_resource.sh
bash run_glue_gazesup_roberta_high_resource.sh
```

To pre-train the Eyettention model (For more details see https://github.com/aeye-lab/Eyettention):
```
python Eyettention_pretrain_CELER.py
```


## Cite our work
If you use our code for your research, please consider citing our paper:

```bibtex
@inproceedings{deng-etal-2024-gazesuplm,
    title = "Fine-Tuning Pre-Trained Language Models with Gaze Supervision",
    author = {Deng, Shuwen  and
      Prasse, Paul  and
      Reich, David  and
      Scheffer, Tobias  and
      J{\"a}ger, Lena},
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "",
    doi = "",
    pages = "",
}

```
