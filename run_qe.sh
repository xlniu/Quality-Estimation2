export QE_DIR=/data/niuxiaolei/learn/pytorch-pretrained-BERT-master/pytorch-pretrained-BERT-master/examples/QE
python run_qe.py \
  --bert_model=./pretrain-models/bert-base-multilingual-cased \
  --do_train \
  --do_eval \
  --data_dir=${QE_DIR} \
  --train_batch_size=32 \
  --eval_batch_size=128 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --hidden_dim=1000 \
  --steps_per_stats=20 \
  --steps_per_eval=100 \
  --max_seq_length=128 \
  --output_dir=./examples/tmp/qe_output/ \
  --gradient_accumulation_steps=1
