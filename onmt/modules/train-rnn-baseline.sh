#!/bin/sh
export LD_LIBRARY_PATH=/home/pquan/local/lib/:/home/pquan/local/lib64/:/usr/local/cuda/lib64/

# For Torch7
. /home/pquan/torch.fresh/install/bin/torch-activate
onmt=/project/mt2017/project/postwmt17/OpenNMT


src=en
tgt=lv

modelType=baseline
SAVEDIR=saves/$src-$tgt
MODELDIR=$SAVEDIR/models/$modelType
LOGDIR=$SAVEDIR/logs

mkdir -p $MODELDIR
mkdir -p $LOGDIR

type=lstm
method=adam
wvsize=512
size=512

length=48
#~ 
th $onmt/bin/train.lua -data $SAVEDIR/tensor-$length-train.t7 -save_model $MODELDIR/model.$type.$method.$size \
			  -layers 2 \
			  -rnn_size $size \
			  -brnn \
			  -brnn_merge concat \
			  -word_vec_size $wvsize \
			  -input_feed 1 \
			  -dropout 0.2 \
			  -optim $method \
			  -learning_rate 0.001 \
			  -max_batch_size 128 \
			  -start_epoch 1 \
			  -end_epoch 5 \
			  -save_every 20000 \
			  -learning_rate_decay 1 -gpuid 1 \
			   2>&1 | tee $LOGDIR/$type.$method.$size.$modelType.bt.log
			   
#~ th $onmt/bin/train.lua -data $SAVEDIR/tensor-$length-train.t7 -save_model $MODELDIR/model.$type.$method.$size \
			  #~ -layers 2 \
			  #~ -rnn_size $size \
			  #~ -brnn \
			  #~ -brnn_merge concat \
			  #~ -word_vec_size $size \
			  #~ -input_feed 1 \
			  #~ -dropout 0.2 \
			  #~ -optim $method \
			  #~ -learning_rate 0.0005 \
			  #~ -max_batch_size 128 \
			  #~ -start_epoch 1 \
			  #~ -end_epoch 5 \
			  #~ -save_every 20000 \
			  #~ -train_from $MODELDIR/model.lstm.adam.1024_checkpoint_epoch5.00_ppl=16.36_bleu=27.57.t7 \
			  #~ -learning_rate_decay 1 -gpuid 1 \
			   #~ 2>&1 | tee $LOGDIR/$type.$method.$size.$modelType.bt.2.log
#~ th $onmt/bin/train.lua -data $SAVEDIR/tensor-$length-train.t7 -save_model $MODELDIR/model.$type.$method.$size \
			  #~ -layers 2 \
			  #~ -rnn_size $size \
			  #~ -brnn \
			  #~ -brnn_merge concat \
			  #~ -word_vec_size $size \
			  #~ -input_feed 1 \
			  #~ -dropout 0.2 \
			  #~ -optim $method \
			  #~ -learning_rate 0.0002 \
			  #~ -max_batch_size 128 \
			  #~ -start_epoch 8 \
			  #~ -end_epoch 9 \
			  #~ -save_every 20000 \
			  #~ -train_from $MODELDIR/model.lstm.adam.1024_checkpoint_epoch2.53_ppl=15.39_bleu=28.50.t7 \
			  #~ -learning_rate_decay 1 -gpuid 1 \
			   #~ 2>&1 | tee $LOGDIR/$type.$method.$size.$modelType.bt.2.log
			   #~ 
			  
			  


