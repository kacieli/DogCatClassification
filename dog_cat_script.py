import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import UnidentifiedImageError, ImageFile
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

root_save_main='/home/li.kaci/'

timenow = datetime.now().strftime('%Y-%m-%d-%H%M%S')
root_save = os.path.join(root_save_main, timenow).replace("\\","/")
if not os.path.exists(root_save):
    os.makedirs(root_save)
os.chdir(root_save)


# Directory where you unzipped your data
train_dir = '/home/li.kaci/dog_cat_balanced_unzipped/dog_cat_cleaned'

# Enable error handling for loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Define dataset paths
batch_size = 16
target_size = (128, 128)  # Reduce image size for faster training

# Image Data Generators
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2  # 80% training, 20% validation
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

layer1 = 64
layer2 = 128
layer3 = 128
layer4 = 128
layer5 = 128

# Build the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(layer1, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Conv2D(layer2, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Conv2D(layer3, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Conv2D(layer4, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Conv2D(layer5, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    
    # Output Layer (Predict breeds; assuming multiple breeds)
    tf.keras.layers.Dense(train_generator.num_classes, activation='softmax')  
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=50,  # Change epochs to a suitable number
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)

# Evaluate the model
val_loss, val_acc = model.evaluate(validation_generator)
print(f"Validation Loss: {val_loss}")
print(f"Validation Accuracy: {val_acc}")
print("Layers: ", layer1, ", ", layer2, ", ", layer3, ", ", layer4, ", ", layer5)

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
print(f"Accuracy: {correct_predictions / total_images * 100:.2f}%")



# Get class indices
class_indices = validation_generator.class_indices
dog_class = class_indices['dog']  # Assuming 'dog' is the key for dog images
cat_class = class_indices['cat']  # Assuming 'cat' is the key for cat images

# Get the true and predicted labels for dogs and cats
correct_dog_predictions = np.sum((y_true == dog_class) & (y_pred_classes == dog_class))
correct_cat_predictions = np.sum((y_true == cat_class) & (y_pred_classes == cat_class))

# Get the total number of dog and cat images
total_dogs = np.sum(y_true == dog_class)
total_cats = np.sum(y_true == cat_class)

# Print results
print(f"Correctly classified dogs: {correct_dog_predictions} out of {total_dogs}")
print(f"Correctly classified cats: {correct_cat_predictions} out of {total_cats}")

# Optionally, calculate accuracy for each class
dog_accuracy = correct_dog_predictions / total_dogs * 100
cat_accuracy = correct_cat_predictions / total_cats * 100

print(f"Dog classification accuracy: {dog_accuracy:.2f}%")
print(f"Cat classification accuracy: {cat_accuracy:.2f}%")


print("True labels:", y_true[:10])
print("Predicted labels:", y_pred_classes[:10])