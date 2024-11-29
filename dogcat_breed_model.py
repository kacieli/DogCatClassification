import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

from PIL import UnidentifiedImageError, ImageFile
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import shutil




# Paths
train_dir = '/home/li.kaci/train_set'  
test_dir = '/home/li.kaci/test_set' 
#dog_dir = os.path.join(base_dir, 'dog')
#cat_dir = os.path.join(base_dir, 'cat')

# Move all breed folders from 'dog' directory to the base directory
#for breed in os.listdir(dog_dir):
#    breed_path = os.path.join(dog_dir, breed)
#    if os.path.isdir(breed_path):
#        shutil.move(breed_path, base_dir)

# Move all breed folders from 'cat' directory to the base directory
#for breed in os.listdir(cat_dir):
#    breed_path = os.path.join(cat_dir, breed)
#    if os.path.isdir(breed_path):
#        shutil.move(breed_path, base_dir)

# Optionally, delete the now-empty 'dog' and 'cat' directories
#os.rmdir(dog_dir)
#os.rmdir(cat_dir)



# Enable error handling for loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Define dataset paths
batch_size = 16
target_size = (128, 128)  # Reduce image size for faster training





# Image Data Generators
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.3  # 80% training, 20% validation
)

# Training and validation generators
train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='categorical',  # 'categorical' because we'll predict multiple breeds
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    train_dir,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)


# Test generator
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(128, 128),
    batch_size=16,
    class_mode='categorical',
    shuffle=False
)

layer1 = 64
layer2 = 128
layer3 = 128
layer4 = 128
layer5 = 128
layer6 = 128
layer7 = 64
#layer8 = 64

# Build the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(layer1, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    tf.keras.layers.BatchNormalization(),
    #tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Conv2D(layer2, (3, 3), activation='relu'),
    #tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Conv2D(layer3, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    #tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Conv2D(layer4, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    #tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.25),
    
    tf.keras.layers.Conv2D(layer5, (3, 3), activation='relu'),
    #tf.keras.layers.BatchNormalization(),
    #tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.25),
    
    tf.keras.layers.Conv2D(layer6, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.25),
    
    tf.keras.layers.Conv2D(layer7, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    #tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.25),

    #tf.keras.layers.Conv2D(layer8, (3, 3), activation='relu'),
    #tf.keras.layers.BatchNormalization(),
    #tf.keras.layers.MaxPooling2D((2, 2)),
    #tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    
    # Output Layer (Predict breeds; assuming multiple breeds)
    tf.keras.layers.Dense(train_generator.num_classes, activation='softmax')  # This will match the number of breeds
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=20,  # Change epochs to a suitable number
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)

#model.save(root_save_main + '/' + str(timenow) + '/dogcat_model1.h5')

# Evaluate the model
val_loss, val_acc = model.evaluate(validation_generator)
print(f"Validation Loss: {val_loss}")
print(f"Validation Accuracy: {val_acc}")
#print("Layers: ", layer1, ", ", layer2, ", ", layer3, ", ", layer4,  ", ", layer5)
print("Layers: ", layer1, ", ", layer2, ", ", layer3, ", ", layer4,  ", ", layer5, ", ", layer6, ", ", layer7)


test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.samples // batch_size)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_acc}")

# Predict on test set
test_steps = int(np.ceil(test_generator.samples / batch_size))
y_test_true = test_generator.classes
y_test_pred = model.predict(test_generator, steps=test_steps)
y_test_pred_classes = np.argmax(y_test_pred, axis=1)


print(f"Unique true labels: {sorted(set(y_test_true))}")
print(f"Unique predicted labels: {sorted(set(y_test_pred_classes))}")


# Confusion Matrix
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

class_labels = list(test_generator.class_indices.keys())


print(f"Number of class labels: {len(class_labels)}")



cm_test = confusion_matrix(y_test_true, y_test_pred_classes)

print(f"Confusion matrix shape: {cm_test.shape}")


if cm_test.shape[0] != len(class_labels):
    print("Mismatch detected! Adjusting labels.")
    class_labels = class_labels[:cm_test.shape[0]]

disp_test = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=class_labels)
disp_test.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix (Test Set)')
plt.show()


root_save_main = '/home/li.kaci/'

timenow = datetime.now().strftime('%Y-%m-%d-%H%M%S')
root_save = os.path.join(root_save_main, timenow).replace("\\", "/")
if not os.path.exists(root_save):
    os.makedirs(root_save)
os.chdir(root_save)


plt.savefig(root_save_main + '/' + str(timenow) + '/test_confusionmatrix.png', bbox_inches='tight')

# Classification Report
print("Test Set Classification Report:")
print(classification_report(y_test_true, y_test_pred_classes, target_names=class_labels))



# Compare true labels with predicted labels
correct_predictions = np.sum(y_test_true == y_test_pred_classes)

# Get the total number of images
total_images = len(y_test_true)

# Print the number of correct classifications and the accuracy
print(f"Correctly classified images: {correct_predictions} out of {total_images}")
print(f"Test Accuracy: {correct_predictions / total_images * 100:.2f}%")

# Get class indices (for each breed within dogs and cats)
class_indices = test_generator.class_indices

# Get the true and predicted labels for each breed
correct_breed_predictions = []
total_breeds = len(class_indices)

for breed_name, breed_index in class_indices.items():
    correct_breed_predictions.append(np.sum((y_test_true == breed_index) & (y_test_pred_classes == breed_index)))
    total_breed_images = np.sum(y_test_true == breed_index)
    breed_accuracy = correct_breed_predictions[-1] / total_breed_images * 100
    print(f"Breed: {breed_name}")
    print(f"Correctly classified {breed_name}: {correct_breed_predictions[-1]} out of {total_breed_images}")
    print(f"Accuracy for {breed_name}: {breed_accuracy:.2f}%")

print("True labels:", y_test_true[:10])
print("Predicted labels:", y_test_pred_classes[:10])



breed_names = []
correct_predictions_per_breed = []

# Populate the lists with breed names and correct predictions
for breed_name, breed_index in class_indices.items():
    correct_count = np.sum((y_test_true == breed_index) & (y_test_pred_classes == breed_index))
    breed_names.append(breed_name)
    correct_predictions_per_breed.append(correct_count)


# Sort the data in descending order
sorted_indices = np.argsort(correct_predictions_per_breed)[::-1]
sorted_breed_names = [breed_names[i] for i in sorted_indices]
sorted_correct_predictions = [correct_predictions_per_breed[i] for i in sorted_indices]

# Create the bar chart
plt.figure(figsize=(16, 8))  # Larger figure size for better readability
plt.bar(sorted_breed_names, sorted_correct_predictions, color='skyblue')

# Add labels and title
plt.xlabel('Breed', fontsize=12)
plt.ylabel('Number of Correctly Classified Images', fontsize=12)
plt.title('Correctly Classified Images Per Breed', fontsize=14)

# Rotate x-axis labels and show only every nth label
plt.xticks(ticks=range(0, len(sorted_breed_names)), labels=sorted_breed_names, 
           rotation=90, ha='center', fontsize=8)  # Rotate labels for clarity

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plot
plt.show()


plt.savefig(root_save_main + '/' + str(timenow) + '/test_correct_class_breeds.png', bbox_inches='tight')






# Plotting
plt.figure(figsize=(12, 5))

# Plot training and validation accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.savefig(root_save_main + '/' + str(timenow) + '/accuracy.png', bbox_inches='tight')

# Plot training and validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.savefig(root_save_main + '/' + str(timenow) + '/loss.png', bbox_inches='tight')

# Evaluate performance with confusion matrix and classification report
validation_steps = int(np.ceil(validation_generator.samples / batch_size))
y_true = validation_generator.classes
y_pred = model.predict(validation_generator, steps=validation_steps)
y_pred_classes = np.argmax(y_pred, axis=1)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=validation_generator.class_indices.keys())
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

plt.savefig(root_save_main + '/' + str(timenow) + '/confusionmatrix.png', bbox_inches='tight')

# Classification Report
print(classification_report(y_true, y_pred_classes, target_names=validation_generator.class_indices.keys()))

# Compare true labels with predicted labels
correct_predictions = np.sum(y_true == y_pred_classes)

# Get the total number of images
total_images = len(y_true)

# Print the number of correct classifications and the accuracy
print(f"Correctly classified images: {correct_predictions} out of {total_images}")
print(f"Val Accuracy: {correct_predictions / total_images * 100:.2f}%")

# Get class indices (for each breed within dogs and cats)
class_indices = validation_generator.class_indices

# Get the true and predicted labels for each breed
correct_breed_predictions = []
total_breeds = len(class_indices)

for breed_name, breed_index in class_indices.items():
    correct_breed_predictions.append(np.sum((y_true == breed_index) & (y_pred_classes == breed_index)))
    total_breed_images = np.sum(y_true == breed_index)
    breed_accuracy = correct_breed_predictions[-1] / total_breed_images * 100
    print(f"Breed: {breed_name}")
    print(f"Correctly classified {breed_name}: {correct_breed_predictions[-1]} out of {total_breed_images}")
    print(f"Accuracy for {breed_name}: {breed_accuracy:.2f}%")

print("True labels:", y_true[:10])
print("Predicted labels:", y_pred_classes[:10])
