root_dir=/path/to/AWD_FINAL
SRC=${root_dir}/src/AWD
CACHE=${root_dir}/CACHE
BERTLR=4e-5
TASK=stsa
NUMEXAMPLES=10
adv_lr=0.01
adv_iter=1
gamma=0.005

fileName=AWD_iter${adv_iter}_advlr${adv_lr}_gamma${gamma}
for i in {0..14};
do
    RAWDATADIR=${root_dir}/datasets/${TASK}/num_train_${NUMEXAMPLES}/exp_${i}
    mkdir ${RAWDATADIR}/${fileName}
    python $SRC/classifier.py --task $TASK  --data_dir $RAWDATADIR --seed ${i} --learning_rate $BERTLR --cache $CACHE --name ${fileName} --adv_lr ${adv_lr} --adv_iter ${adv_iter} --gamma ${gamma} > $RAWDATADIR/${fileName}/log.txt

done
python ${root_dir}/src/utils/calc_mean.py --dir ${root_dir}/datasets/${TASK}/num_train_${NUMEXAMPLES}/${fileName}.log
