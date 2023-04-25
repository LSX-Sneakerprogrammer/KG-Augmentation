# Data Augmented Knowledge Graph Completion via Pre-trained Language Models

The repository is modified from [KG-BERT](https://github.com/yao8839836/kg-bert) and tested on Python 3.7+.

The repository is partially based on [huggingface transformers](https://github.com/huggingface/transformers), [KG-BERT](https://github.com/yao8839836/kg-bert) and [AnyBURL](https://web.informatik.uni-mannheim.de/AnyBURL/).


## Installing requirement packages

```bash
pip install -r requirements.txt
```

## Data

(1) The benchmark knowledge graph datasets are in ./data. 

(2) The demo dataset in ./demo_data can help run small demo datasets. 

(3) entity2text.txt or entity2textlong.txt in each dataset contains entity textual sequences.

(4) relation2text.txt in each dataset contains relation textual sequences.

## Extract augmented data

### Extract multi-hop facts

Here is an example of extracting 2-hop facts for demo UMLS dataset

```shell
python3 extract_multihop_triples.py 
    --data_dir ./demo_data/umls 
    --data_type multihop
    --K 2
    --T 10
```

### Extract implicit facts

Here is an example of extracting implicit facts for demo UMLS dataset. The steps are as follows:

```shell
python3 extract_implicit_triples.py 
    --data_dir ./demo_data/umls 
    --data_type multihop
    --cs 0.85
    --n_body 100
    --T 1
```

<!-- (1) Run anyburl model under ./anyburl path, by the steps in [AnyBURL](https://web.informatik.uni-mannheim.de/AnyBURL/).

(2) For filtering and extracting from rules, run ./anyburl/extract_rules.py (Currently, some paths in the file need to be modified into parser form) -->


## Run for original KG-BERT
 

### WN18RR

```shell
python3 run_bert_link_prediction.py
--task_name kg  
--do_train  
--do_eval 
--do_predict 
--data_dir ./demo_data/WN18RR
--bert_model bert-base-uncased
--max_seq_length 50
--train_batch_size 32 
--learning_rate 5e-5 
--num_train_epochs 5.0 
--output_dir ./output_WN18RR/  
--gradient_accumulation_steps 1 
--eval_batch_size 5000
```

### UMLS

```shell
python3 run_bert_link_prediction.py
--task_name kg  
--do_train  
--do_eval 
--do_predict 
--data_dir ./demo_data/umls
--bert_model bert-base-uncased
--max_seq_length 15
--train_batch_size 32 
--learning_rate 5e-5 
--num_train_epochs 5.0 
--output_dir ./output_umls/  
--gradient_accumulation_steps 1 
--eval_batch_size 135
```

### FB15k-237

```shell
python3 run_bert_link_prediction.py
--task_name kg  
--do_train  
--do_eval 
--do_predict 
--data_dir ./demo_data/FB15k-237
--bert_model bert-base-uncased
--max_seq_length 150
--train_batch_size 32 
--learning_rate 5e-5 
--num_train_epochs 5.0 
--output_dir ./output_FB15k-237/  
--gradient_accumulation_steps 1 
--eval_batch_size 1500
```

### NELL-ONE

```shell
python3 run_bert_link_prediction.py 
    --task_name kg  
    --do_train  
    --do_eval 
    --do_predict 
    --data_dir ./demo_data/NELL_ONE_reconstructed
    --bert_model bert-base-uncased
    --max_seq_length 32 
    --train_batch_size 32 
    --learning_rate 5e-5 
    --num_train_epochs 5.0 
    --output_dir ./output_NELL-ONE/  
    # --load_weights_path $load_path \
    --gradient_accumulation_steps 1 
    --eval_batch_size 5000 
```

## Run for KG-BERT via multi-hop augmentation
 

### WN18RR

```shell
python3 run_bert_link_prediction_multi_hop.py 
    --task_name kg  
    --do_train
    --do_extend_train
    --do_eval 
    --do_predict 
    --data_dir ./demo_data/WN18RR 
    --bert_model bert-base-uncased
    # --load_weights_path $load_path # if needed for relaod model weights from .h5 file
    --max_seq_length 60 
    --train_batch_size 32 
    --learning_rate 5e-5 
    --num_train_epochs 5.0 
    --output_dir ./output_WN18RR_multi-hop/ 
    --gradient_accumulation_steps 1 
    --eval_batch_size 4500 
```

### UMLS

```shell
python3 run_bert_link_prediction_multi_hop.py 
    --task_name kg  
    --do_train
    --do_extend_train
    --do_eval 
    --do_predict 
    # --load_weights_path $load_path
    --data_dir ./demo_data/umls 
    --bert_model bert-base-uncased
    --max_seq_length 20 
    --train_batch_size 32 
    --learning_rate 5e-5 
    --num_train_epochs 5.0 
    --output_dir ./output_UMLS_multi-hop/ 
    --gradient_accumulation_steps 1 
    --eval_batch_size 135 
```

### FB15k-237 (few-shot learning)

```shell
python3 run_bert_link_prediction_multi_hop.py \
    --task_name kg  
    --do_train
    --do_extend_train
    --do_eval 
    --do_predict 
    --data_dir ./demo_data/FB15k-237 
    --bert_model bert-base-uncased 
    --max_seq_length 150 
    --train_batch_size 32 
    --learning_rate 5e-5 
    --num_train_epochs 5.0 
    --output_dir ./output_FB15k-237_multi-hop/  
    # --load_weights_path $load_path 
    --gradient_accumulation_steps 1 
    --eval_batch_size 1500 
```

### NELL-ONE reconstructed (few-shot learning)

```shell
python3 run_bert_link_prediction_multi_hop.py \
    --task_name kg  
    --do_train
    --do_extend_train
    --do_eval 
    --do_predict 
    --data_dir ./demo_data/NELL_ONE_reconstructed
    # --load_weights_path $load_path 
    --bert_model bert-base-uncased 
    --max_seq_length 40 
    --train_batch_size 32 
    --learning_rate 5e-5 
    --num_train_epochs 5.0 
    --output_dir ./output_NELL-ONE_multi-hop/ 
    --gradient_accumulation_steps 1 
    --eval_batch_size 5000 
```

## Run for KG-BERT via rule based augmentation

### WN18RR

```shell
python3 run_bert_link_prediction_dropout.py \
    --task_name kg  
    --do_train
    --do_eval 
    --do_predict 
    --data_dir ./demo_data/WN18RR 
    --bert_model bert-base-uncased 
    # --load_weights_path $load_path 
    --max_seq_length 50 
    --train_batch_size 32 
    --learning_rate 5e-5 
    --num_train_epochs 5.0 
    --output_dir ./output_WN18RR_dropout/ 
    --gradient_accumulation_steps 1 
    --eval_batch_size 5000 
```

### UMLS

```shell
python3 run_bert_link_prediction_dropout.py 
    --task_name kg  
    --do_train
    --do_eval 
    --do_predict 
    # --load_weights_path $load_path
    --data_dir ./demo_data/umls 
    --bert_model bert-base-uncased
    --max_seq_length 15
    --train_batch_size 32 
    --learning_rate 5e-5 
    --num_train_epochs 5.0 
    --output_dir ./output_UMLS_dropout/ 
    --gradient_accumulation_steps 1 
    --eval_batch_size 135 
```

## Reproducing results for KG-BERT via both two augmentations

### WN18RR

```shell
python3 run_bert_link_prediction_mixed.py \
    --task_name kg  
    --do_train
    --do_extend_train
    --do_eval 
    --do_predict 
    --data_dir ./demo_data/WN18RR 
    --bert_model bert-base-uncased 
    # --load_weights_path $load_path 
    --max_seq_length 50 
    --train_batch_size 32 
    --learning_rate 5e-5 
    --num_train_epochs 5.0 
    --output_dir ./output_WN18RR_mixed/ 
    --gradient_accumulation_steps 1 
    --eval_batch_size 5000 
```

### UMLS

```shell
python3 run_bert_link_prediction_mixed.py 
    --task_name kg  
    --do_train
    --do_eval 
    --do_predict 
    # --load_weights_path $load_path
    --data_dir ./demo_data/umls 
    --bert_model bert-base-uncased
    --max_seq_length 20
    --train_batch_size 32 
    --learning_rate 5e-5 
    --num_train_epochs 5.0 
    --output_dir ./output_UMLS_mixed/ 
    --gradient_accumulation_steps 1 
    --eval_batch_size 135 
```
