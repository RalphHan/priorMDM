for i in {0..7}
do
    CUDA_VISIBLE_DEVICES=$i python ik_all.py $i 8 &
done
wait
