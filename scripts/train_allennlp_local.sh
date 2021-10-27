# Run allennlp training locally

#
# edit these variables before running script
DATASET='citation_intent' # 'sci-cite' 'citation_intent'
TASK='text_classification'
with_finetuning='_finetune' #'_finetune'  # or '' for not fine tuning
with_multitask='' #'_multitask'  # or '' for not multitask
dataset_size=1688

export BERT_VOCAB=scibert_scivocab_uncased/vocab.txt
export BERT_WEIGHTS=scibert_scivocab_uncased/weights.tar.gz

export DATASET_SIZE=$dataset_size

CONFIG_FILE=allennlp_config/"$TASK""$with_finetuning""$with_multitask".json

SEED=465327987
PYTORCH_SEED=`expr $SEED / 10`
NUMPY_SEED=`expr $PYTORCH_SEED / 10`
export SEED=$SEED
export PYTORCH_SEED=$PYTORCH_SEED
export NUMPY_SEED=$NUMPY_SEED

export IS_LOWERCASE=true
export TRAIN_PATH=data/$TASK/$DATASET/train.txt
export DEV_PATH=data/$TASK/$DATASET/dev.txt
export TEST_PATH=data/$TASK/$DATASET/test.txt
export AUX_PATH=data/$TASK/$DATASET/scaffolds/sections-scaffold-train.jsonl
export AUX_2_PATH=data/$TASK/$DATASET/scaffolds/cite-worthiness-scaffold-train.jsonl

export CUDA_DEVICE=0

export GRAD_ACCUM_BATCH_SIZE=32
export NUM_EPOCHS=4
export LEARNING_RATE=0.00003

#python scibert/training/train_local.py train_multitask_2 $CONFIG_FILE  --include-package scibert -s "$@"
python -m allennlp.run train $CONFIG_FILE  --include-package scibert -s "$@"
