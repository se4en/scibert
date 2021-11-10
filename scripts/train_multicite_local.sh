# Run allennlp training locally

#
# edit these variables before running script
DATASET='multicite'
TASK='text_classification'
with_finetuning='_finetune' #'_finetune'  # or '' for not fine tuning
with_multitask='' #'_multitask'  # or '' for not multitask
with_multilabel='_multilabel' #'_multilabel'  # or '' for not multilabel
dataset_size=10000

export BERT_VOCAB=scibert_scivocab_uncased/vocab.txt
export BERT_WEIGHTS=scibert_scivocab_uncased/weights.tar.gz

export DATASET_SIZE=$dataset_size

CONFIG_FILE=allennlp_config/"$TASK""$with_finetuning""$with_multitask""$with_multilabel".json

SEED=6734832
PYTORCH_SEED=`expr $SEED / 10`
NUMPY_SEED=`expr $PYTORCH_SEED / 10`
export SEED=$SEED
export PYTORCH_SEED=$PYTORCH_SEED
export NUMPY_SEED=$NUMPY_SEED

export IS_LOWERCASE=true
export TRAIN_PATH=data/$TASK/$DATASET/train.jsonl
export DEV_PATH=data/$TASK/$DATASET/dev.jsonl
export TEST_PATH=data/$TASK/$DATASET/test.jsonl
# export AUX_PATH=data/$TASK/$DATASET/scaffolds/cite-sections.jsonl
# export AUX_2_PATH=data/$TASK/$DATASET/scaffolds/cite-worthiness.jsonl

export CUDA_DEVICE=-1

export GRAD_ACCUM_BATCH_SIZE=32
export NUM_EPOCHS=5
export LEARNING_RATE=0.0001

python -m allennlp.run train $CONFIG_FILE  --include-package scibert -s "$@"