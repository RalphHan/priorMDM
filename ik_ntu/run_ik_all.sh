for i in {0..7}
do
    CUDA_VISIBLE_DEVICES=$i python -m ik_ntu.ik $i 8 &
done
wait
