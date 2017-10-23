# Convert the Dataset into the required format(break the train+test data files into separate files) required by the CNN-code

#store train and test image ids
train_image_ids = []
test_image_ids = []
    
with open('data/train_test_split.txt') as train_test_file:
    for image_id,line in enumerate(train_test_file):
        is_train = int(line.strip().split()[1])
        if is_train == 1:
            train_image_ids.append(image_id+1)
        else:
            test_image_ids.append(image_id+1)

print(train_image_ids[0])
print(len(train_image_ids))
print(len(test_image_ids))


#Splitting Images into two files -- images_train.txt and images_test.txt
with open('data/images.txt') as image_locations, open('data/images_test.txt', "w") as test_image_locations, open('data/images_train.txt', "w") as train_image_locations:
    for image_id, line in enumerate(image_locations):
        #print(image_id+1,line.strip().split()[1],)
        if image_id+1 in train_image_ids:
            train_image_locations.write(line)
        else:
            test_image_locations.write(line)



#Splitting Bounding Boxes into two files -- train_labels.txt and test_labels.txt
with open('data/bounding_boxes.txt') as image_locations, open('data/bounding_boxes_test.txt', "w") as test_image_bounding_boxes, open('data/bounding_boxes_train.txt', "w") as train_image_bounding_boxes:
    for image_id, line in enumerate(image_locations):
        #print(image_id+1,line.strip().split()[1],)
        if image_id+1 in train_image_ids:
            train_image_bounding_boxes.write(line)
        else:
            test_image_bounding_boxes.write(line)


#Splitting Labels into two files -- train_labels.txt and test_labels.txt
with open('data/image_class_labels.txt') as image_locations, open('data/image_class_labels_test.txt', "w") as test_image_class, open('data/image_class_labels_train.txt', "w") as train_image_class:
    for image_id, line in enumerate(image_locations):
        #print(image_id+1,line.strip().split()[1],)
        if image_id+1 in train_image_ids:
            train_image_class.write(line)
        else:
            test_image_class.write(line)
