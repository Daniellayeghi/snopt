
MNIST="--result-dir result/mnist --problem mnist   --batch-size 128 --epoch 5 "

if [ "$1" == mnist ]; then
    python3 main_img_clf.py $MNIST --optimizer Adam  --lr 0.001 --seed 0
    python3 main_img_clf.py $MNIST --optimizer SNOpt --lr 0.02 --snopt-eps 0.05 --snopt-freq 100 --seed 0
    python3 main_img_clf.py $MNIST --optimizer SGD   --lr 0.02 --momentum 0.9 --seed 0
fi
