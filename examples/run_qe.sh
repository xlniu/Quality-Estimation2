export QE_DIR=./examples/QE
python run_swag.py \
  --bert_model=../pretrain-models/bert-base-uncased \
  --do_train \
  --do_lower_case \
  --do_eval \
  --data_dir=${QE_DIR} \
  --train_batch_size=32 \
  --eval_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --max_seq_length=140 \
  --output_dir=/tmp/qe_output/ \
  --gradient_accumulation_steps=4
