set -e
for alpha in "0" "1"
do
        for l1 in "0" "0.5" "1"
        do
                python train.py "$alpha" "$l1"
        done
done
