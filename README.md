# bird_classification
Fine-grained classification of bird images from Caltech-UCSD Birds-200-2011 dataset
The required-dataset could be found at [Caltech-UCSD Webpage](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) for the same.

Place all the Downloaded Data in the "/data" folder and run "convert.py" (python convert.py). This will convert the data in the required format i.e. the train and test file data (labels, image-locations and bounding boxes) will be placed in separate files.

Now a sample command to run the (main)code:
python main.py  --batch_size 8 --val_ratio 0.1 --num_epochs 5000 --gpu_mem_frac 0.5 --model vgg

This will create a csv file which will store the prediction for each test-image indexed by its id.

TODO: Use Predictions and Test_labels to show accuray on test-data
