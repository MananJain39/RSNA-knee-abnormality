# B4 — ResNet34d + pool head, 24 slices per series, seed 1

**Question:** does seeing more slices per series (24 instead of 16 of the 32 stored) help?
**Command:** `python scripts/train.py --backbone resnet34d.ra2_in1k --head pool --lr 3e-4 --head-lr 1e-3 --fold f --folds 4 --epochs 10 --bs 6 --n-slices 24 --seed 1 --out /data/exp/B4_resnet34d_24sl`
Everything else as B1.

**Answer: no.** OOF 0.885 (B1: 0.886); gold-58 ensemble 0.901 (B1: 0.905); single folds 0.880–0.896. 50% more compute
for no gain → 16 evenly spaced slices already cover the knee. Still useful as an ensemble member (different seed
and sampling): B1+B4 OOF 0.891. → `results.md`.
