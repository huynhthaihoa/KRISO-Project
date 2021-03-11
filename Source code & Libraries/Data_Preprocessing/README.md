# Data Preprocessing
This library contains some Python scripts that can be used for data preprocessing. The Python version >= 3.0 is required. Each script is described as below:
## voc_to_yolo.py
This script converts annotation files from Pascal VOC format (.xml) to standard YOLO format (.txt):
<p align="center"><img src="docs\convert.png" border="0" height="400"/></p>

### Usage:
Run this script by the command below:
``` bat
python voc_to_yolo.py -p [Directory contains input Pascal VOC annotation files] -c [The link to the file contains class list] -y [Directory contains output YOLO annotation files] 
```
Example:
``` bat
python voc_to_yolo.py -y D:\Projects\Tool\Yolo -p D:\Projects\Tool\Coco -c D:\Projects\Tool\obj.names
```
After running this script, each YOLO annotation file will be generated based on its corresponding Pascal VOC annotation file.
### Benefit:
Since different detection models may require different label annotation format, this script can save time for data labeling.
## yolo_to_voc.py
Converting annotation files from standard YOLO format (.txt) to Pascal VOC format (.xml)
### Usage:
From the folder contains this script, open the Command Prompt and type the command below:
``` bat
python voc_to_yolo.py -y [Directory contains input YOLO annotation files] -c [The link to to the file contains class list] -p [Directory contains output Pascal VOC annotation files]
```
Example:
``` bat
python voc_to_yolo.py -p D:\Projects\Tool\Coco -c D:\Projects\Tool\obj.names -y D:\Projects\Tool\Yolo
```
After running this script, each YOLO annotation file will be generated based on its corresponding Pascal VOC annotation file.
### Benefit:
Since different detection models may require different label annotation format, this script can save time for data labeling.
## generate_dataset_from_voc.py
On each input image with its corresponding Pascal VOC annotation file, this script crops every object bounding box from the image to allocate into the corresponding class directory:
<p align="center"><img src="docs\generate_data.jpg" border="0" height="400"/></p>

### Usage:
From the folder contains this script, open the Command Prompt and type the command below:
``` bat
python generate_dataset_from_voc.py -i [Directory contains input images] -a [Directory contains VOC Pascal annotation files (.xml)] -o [Directory contains object dataset]
```
Example:
``` bat
python generate_dataset_from_voc.py -p D:\Projects\Tool\Image -c D:\Projects\Tool\VOC -y D:\Projects\Tool\Labels
```
### Benefit:
This script can help to check the detection and classification accuracy on each object region from the image.
## generate_dataset_from_yolo.py
On each input image with its corresponding YOLO annotation file, this script crops every object bounding box from the image to allocate into the corresponding class directory:
<p align="center"><img src="docs\generate_data.jpg" border="0" height="400"/></p>

### Usage:
From the folder contains this script, open the Command Prompt and type the command below:
``` bat
python generate_dataset_from_yolo.py -i [Directory contains input images] -a [Directory contains Yolo annotation files (.txt)] -o [Directory contains object dataset]
```
Example:
``` bat
python generate_dataset_from_voc.py -p D:\Projects\Tool\Image -c D:\Projects\Tool\Yolo -y D:\Projects\Tool\Labels
```
### Benefit:
This script can help to check the detection and classification accuracy on each object region from the image.
## update_yolo.py
This script updates the annotation files based on the class directory of each bounding box images.
