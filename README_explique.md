This projects aims at makine xplique attributions method and metrics
works with tensorflow methods for the detection task

### Installation

- First follow README.md installation guide.
- Install xplique master branch


    pip install --upgrade https://github.com/deel-ai/xplique/archive/refs/heads/master.zip


- Install jupyter notebook if you want to test the notebook branch

### Retrain on Lard
- install the additionnal dep:

pip install glob2
pip install pycocotools

- Export lard with the export script :

```
  dataset.export(output_dir ="data/newYoloFormat/", 
               bbx_format="xywh", # Options are 'tlbr', 'tlwh', 'xywh', 'corners'
               normalized=True, 
               label_file="multiple", # 'multiple' produces 1 file per label, as expected by yolo architectures. 
               crop=True, # 'True' recommended to remove the watermark. Pay attention to not crop a picture multiple times
               sep=' ', # Separator in the label file.
               header=False, # 'False' is recommender for multiple files, 'True' for single files. It adds a header with column names in the first line of the labels file  
               ext="txt")
```
- Run keras_cv import script :

```
 python coco_train_script.py --train_images 
 C:\Users\maxime.carrere\PycharmProjects\datasprint\data\newYoloFormat\train\images 
 --train_labels C:\Users\maxime.carrere\PycharmProjects\datasprint\data\newYoloFormat\train\labels 
 --test_images C:\Users\maxime.carrere\PycharmProjects\datasprint\data\newYoloFormat\test\images 
 --test_labels C:\Users\maxime.carrere\PycharmProjects\datasprint\data\newYoloFormat\test\labels 
 --bbox_source_format cxcywh 
 -s coco
```

- run train :
python coco_train_script.py --data_name coco.json
### Run
Run the notebook xplique_detection_notebook.
Alternatively -I did not document it yet-, you can use it with xplique_test.py (parameters to modifity to test
others methods, metrics, models, are at the end).

- additionnal:
windows 10 natif :
  conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
# Anything above 2.10 is not supported on the GPU on Windows Native
python -m pip install "tensorflow<2.11"
# Verify install:
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
