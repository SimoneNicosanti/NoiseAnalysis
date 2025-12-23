python DatasetBuild.py --family yolo11 --variant n-det --batch 5 --gpu --dataset coco128 --layers-num 12 --dataset-size 3_000 --calib-size 100 --eval-size 20
python DatasetBuild.py --family yolo11 --variant n-seg --batch 5 --gpu --dataset coco128 --layers-num 12 --dataset-size 3_000 --calib-size 100 --eval-size 20

python DatasetBuild.py --family yolo11 --variant m-det --batch 5 --gpu --dataset coco128 --layers-num 12 --dataset-size 3_000 --calib-size 100 --eval-size 20
python DatasetBuild.py --family yolo11 --variant m-seg --batch 5 --gpu --dataset coco128 --layers-num 12 --dataset-size 3_000 --calib-size 100 --eval-size 20

python DatasetBuild.py --family yolo11 --variant x-det --batch 5 --gpu --dataset coco128 --layers-num 12 --dataset-size 3_000 --calib-size 100 --eval-size 20
python DatasetBuild.py --family yolo11 --variant x-seg --batch 5 --gpu --dataset coco128 --layers-num 12 --dataset-size 3_000 --calib-size 100 --eval-size 20