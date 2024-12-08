import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

from PIL import UnidentifiedImageError, ImageFile
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import os
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import shutil
import seaborn as sns





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

# Build the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(layer1, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    tf.keras.layers.BatchNormalization(),
    #tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Conv2D(layer2, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Conv2D(layer3, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    #tf.keras.layers.MaxPooling2D((2, 2)),
    #tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Conv2D(layer4, (3, 3), activation='relu'),
    #tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.5),
    
    tf.keras.layers.Conv2D(layer5, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    #tf.keras.layers.MaxPooling2D((2, 2)),
    #tf.keras.layers.Dropout(0.5),
    
    tf.keras.layers.Conv2D(layer6, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.5),
    
    tf.keras.layers.Conv2D(layer7, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
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




plt.figure(figsize=(20, 20))  # Increase the figure size
disp_test = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=class_labels)
disp_test.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')  # Rotate x-axis labels
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


cat_breeds = ['Abyssinian', 'American Bobtail', 'American Curl', 'American Shorthair', 'American Wirehair', 'Applehead Siamese', 'Balinese', 'Bengal', 'Birman', 'Bombay', 'British Shorthair', 'Burmese', 'Burmilla', 'Calico', 'Canadian Hairless', 'Chartreux', 'Chausie', 'Chinchilla', 'Cornish Rex', 'Cymric', 'Devon Rex', 'Dilute Calico', 'Dilute Tortoiseshell', 'Domestic Long Hair', 'Domestic Medium Hair', 'Domestic Short Hair', 'Egyptian Mau', 'Exotic Shorthair', 'Extra-Toes Cat - Hemingway Polydactyl', 'Havana', 'Himalayan', 'Japanese Bobtail', 'Javanese', 'Korat', 'LaPerm', 'Maine Coon', 'Manx', 'Munchkin', 'Nebelung', 'Norwegian Forest Cat', 'Ocicat', 'Oriental Long Hair', 'Oriental Short Hair', 'Oriental Tabby', 'Persian', 'Pixiebob', 'Ragamuffin', 'Ragdoll', 'Russian Blue', 'Scottish Fold', 'Selkirk Rex', 'Siamese', 'Siberian', 'Silver', 'Singapura', 'Snowshoe', 'Somali', 'Sphynx - Hairless Cat', 'Tabby', 'Tiger', 'Tonkinese', 'Torbie', 'Tortoiseshell', 'Turkish Angora', 'Turkish Van', 'Tuxedo', 'York Chocolate']

cat_breeds_set = set(cat_breeds)

# Classify breeds
breed_classification = {
    breed_name: "cat" if breed_name in cat_breeds_set else "dog"
    for breed_name in breed_names
}


# Sort data by correct predictions in descending order
sorted_indices = np.argsort(correct_predictions_per_breed)[::-1]
sorted_breed_names = [breed_names[i] for i in sorted_indices]
sorted_correct_predictions = [correct_predictions_per_breed[i] for i in sorted_indices]

# Assign bar colors
bar_colors = ['blue' if breed_classification[breed] == "cat" else 'orange' for breed in sorted_breed_names]

# Plot the bar chart
plt.figure(figsize=(16, 8))
plt.bar(sorted_breed_names, sorted_correct_predictions, color=bar_colors)

plt.xlabel('Breed', fontsize=12)
plt.ylabel('Number of Correctly Classified Images', fontsize=12)
plt.title('Correctly Classified Images Per Breed', fontsize=14)
plt.xticks(rotation=90, ha='center', fontsize=8)
legend_elements = [
    Patch(facecolor='blue', label='Cat'),
    Patch(facecolor='orange', label='Dog')
]
plt.legend(handles=legend_elements, loc='upper right', fontsize=10)


plt.tight_layout()
plt.show()

plt.savefig(root_save_main + '/' + str(timenow) + '/test_correct_class_breeds.png', bbox_inches='tight')





misclassifications = np.sum(cm_test, axis=1) - np.diag(cm_test)
top_classes = np.argsort(misclassifications)[-10:]  # Top 10 misclassified classes

# Filter confusion matrix
cm_filtered = cm_test[top_classes][:, top_classes]
filtered_labels = [class_labels[i] for i in top_classes]

# Plot the filtered confusion matrix
plt.figure(figsize=(10, 10))
disp_filtered = ConfusionMatrixDisplay(confusion_matrix=cm_filtered, display_labels=filtered_labels)
disp_filtered.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
plt.title('Filtered Confusion Matrix (Top Misclassified Classes)')
plt.show()

plt.savefig(root_save_main + '/' + str(timenow) + '/test_most_misclass_confmtrx.png', bbox_inches='tight')







plt.figure(figsize=(20, 20))
sns.heatmap(cm_test, annot=False, cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix Heatmap')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.show()

plt.savefig(root_save_main + '/' + str(timenow) + '/test_cm_heatmap.png', bbox_inches='tight')


"""
breed_1 = "Balinese"  # Replace with the actual breed name
breed_2 = "Siamese"  # Replace with the actual breed name

# Get the class indices for the specified breeds
class_indices = test_generator.class_indices
breed_1_index = class_indices.get(breed_1)
breed_2_index = class_indices.get(breed_2)


selected_indices = [breed_1_index, breed_2_index]

# Filter both true and predicted labels together
y_test_true_filtered, y_test_pred_filtered = [], []

for true, pred in zip(y_test_true, y_test_pred_classes):
    if true in selected_indices:
        y_test_true_filtered.append(true)
        if pred in selected_indices:
            y_test_pred_filtered.append(pred)
            
            
print("y_test_true_filtered: ", y_test_true_filtered)
print("y_test_pred_filtered: ", y_test_pred_filtered)

# Ensure consistent lengths
#if len(y_test_true_filtered) != len(y_test_pred_filtered):
#    raise ValueError("Mismatch in filtered true and predicted label lengths. Check filtering logic.")

# Remap labels
mapping = {breed_1_index: 0, breed_2_index: 1}
y_test_true_filtered = [mapping[label] for label in y_test_true_filtered]
y_test_pred_filtered = [mapping[label] for label in y_test_pred_filtered]

if len(y_test_true_filtered) != len(y_test_pred_filtered):
    print("Warning: Lengths do not match after mapping.")
    print(f"True labels: {y_test_true_filtered}")
    print(f"Predicted labels: {y_test_pred_filtered}")

# Generate the confusion matrix
cm_filtered = confusion_matrix(y_test_true_filtered, y_test_pred_filtered)

# Display the confusion matrix
disp_filtered = ConfusionMatrixDisplay(confusion_matrix=cm_filtered, display_labels=[breed_1, breed_2])
disp_filtered.plot(cmap=plt.cm.Blues)
plt.title(f'Confusion Matrix for {breed_1} and {breed_2}')
plt.show()

# Save the confusion matrix plot
plt.savefig(root_save_main + '/' + str(timenow) + f'/test_confusionmatrix_{breed_1}_{breed_2}.png', bbox_inches='tight')

# Generate and display the classification report
print(f"Classification Report for {breed_1} and {breed_2}:")
print(classification_report(y_test_true_filtered, y_test_pred_filtered, target_names=[breed_1, breed_2]))




# Specify the two breeds to compare
selected_breeds = ['Siberian_husky', 'malamute']

# Get indices for the selected breeds
#breed_indices = {breed: index for breed, index in test_generator.class_indices.items() if breed in selected_breeds}

if len(breed_indices) != 2:
    raise ValueError("Please ensure the selected breeds exist in the dataset and exactly two are selected.")

# Map breed names to indices
breedA_idx, breedB_idx = breed_indices[selected_breeds[0]], breed_indices[selected_breeds[1]]

# Filter the true and predicted labels for the selected breeds
breed_filter = np.isin(y_test_true, [breedA_idx, breedB_idx])
y_test_filtered_true = y_test_true[breed_filter]
y_test_filtered_pred = y_test_pred_classes[breed_filter]

# Remap the indices to 0 and 1 for the two breeds
remap = {breedA_idx: 0, breedB_idx: 1}
y_test_filtered_true = np.array([remap[label] for label in y_test_filtered_true])
y_test_filtered_pred = np.array([remap[label] for label in y_test_filtered_pred])

# Create the confusion matrix
cm_filtered = confusion_matrix(y_test_filtered_true, y_test_filtered_pred)

# Display the confusion matrix
disp_filtered = ConfusionMatrixDisplay(confusion_matrix=cm_filtered, display_labels=selected_breeds)
disp_filtered.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix for Balinese and Siamese Cat')
plt.show()

plt.savefig(root_save_main + '/' + str(timenow) + '/test_confusionmatrix_balinese_siamese.png', bbox_inches='tight')


"""

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





breed_names = []
correct_predictions_per_breed = []

# Populate the lists with breed names and correct predictions
for breed_name, breed_index in class_indices.items():
    correct_count = np.sum((y_true == breed_index) & (y_pred_classes == breed_index))
    breed_names.append(breed_name)
    correct_predictions_per_breed.append(correct_count)



# Classify breeds
breed_classification = {
    breed_name: "cat" if breed_name in cat_breeds_set else "dog"
    for breed_name in breed_names
}


# Sort data by correct predictions in descending order
sorted_indices = np.argsort(correct_predictions_per_breed)[::-1]
sorted_breed_names = [breed_names[i] for i in sorted_indices]
sorted_correct_predictions = [correct_predictions_per_breed[i] for i in sorted_indices]

# Assign bar colors
bar_colors = ['blue' if breed_classification[breed] == "cat" else 'orange' for breed in sorted_breed_names]

# Plot the bar chart
plt.figure(figsize=(16, 8))
plt.bar(sorted_breed_names, sorted_correct_predictions, color=bar_colors)

plt.xlabel('Breed', fontsize=12)
plt.ylabel('Number of Correctly Classified Images', fontsize=12)
plt.title('Correctly Classified Images Per Breed', fontsize=14)
plt.xticks(rotation=90, ha='center', fontsize=8)
plt.legend(["Cat", "Dog"], loc='upper right', fontsize=10)

plt.tight_layout()
plt.show()

plt.savefig(root_save_main + '/' + str(timenow) + '/val_correct_class_breeds.png', bbox_inches='tight')



breed_names = []
correct_predictions_per_breed = []

# Populate the lists with breed names and correct predictions
for breed_name, breed_index in class_indices.items():
    correct_count = np.sum((y_test_true == breed_index) & (y_test_pred_classes == breed_index))
    breed_names.append(breed_name)
    correct_predictions_per_breed.append(correct_count)

# Create the bar chart
plt.figure(figsize=(12, 6))
plt.bar(breed_names, correct_predictions_per_breed, color='skyblue')

# Add labels and title
plt.xlabel('Breed', fontsize=12)
plt.ylabel('Number of Correctly Classified Images', fontsize=12)
plt.title('Correctly Classified Images Per Breed', fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=10)  # Rotate breed names for better readability
plt.legend(["Cat", "Dog"], loc='upper right', fontsize=10)
plt.tight_layout()  # Adjust layout to prevent overlap

# Show the plot
plt.show()

plt.savefig(root_save_main + '/' + str(timenow) + '/val_correct_class_breeds.png', bbox_inches='tight')




misclassifications = np.sum(cm_test, axis=1) - np.diag(cm_test)
top_classes = np.argsort(misclassifications)[-10:]  # Top 10 misclassified classes

# Filter confusion matrix
cm_filtered = cm_test[top_classes][:, top_classes]
filtered_labels = [class_labels[i] for i in top_classes]

# Plot the filtered confusion matrix
plt.figure(figsize=(10, 10))
disp_filtered = ConfusionMatrixDisplay(confusion_matrix=cm_filtered, display_labels=filtered_labels)
disp_filtered.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
plt.title('Filtered Confusion Matrix (Top Misclassified Classes)')
plt.show()

plt.savefig(root_save_main + '/' + str(timenow) + '/val_most_misclass_confmtrx.png', bbox_inches='tight')







plt.figure(figsize=(20, 20))
sns.heatmap(cm_test, annot=False, cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix Heatmap')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.show()

plt.savefig(root_save_main + '/' + str(timenow) + '/val_cm_heatmap.png', bbox_inches='tight')




breed_names = []
correct_predictions_per_breed = []

# Populate the lists with breed names and correct predictions
for breed_name, breed_index in class_indices.items():
    correct_count = np.sum((y_true == breed_index) & (y_pred_classes == breed_index))
    breed_names.append(breed_name)
    correct_predictions_per_breed.append(correct_count)




# Classify breeds
breed_classification = {
    breed_name: "cat" if breed_name in cat_breeds_set else "dog"
    for breed_name in breed_names
}


# Sort data by correct predictions in descending order
sorted_indices = np.argsort(correct_predictions_per_breed)[::-1]
sorted_breed_names = [breed_names[i] for i in sorted_indices]
sorted_correct_predictions = [correct_predictions_per_breed[i] for i in sorted_indices]

# Assign bar colors
bar_colors = ['blue' if breed_classification[breed] == "cat" else 'orange' for breed in sorted_breed_names]

# Plot the bar chart
plt.figure(figsize=(16, 8))
plt.bar(sorted_breed_names, sorted_correct_predictions, color=bar_colors)

plt.xlabel('Breed', fontsize=12)
plt.ylabel('Number of Correctly Classified Images', fontsize=12)
plt.title('Correctly Classified Images Per Breed', fontsize=14)
plt.xticks(rotation=90, ha='center', fontsize=8)
legend_elements = [
    Patch(facecolor='blue', label='Cat'),
    Patch(facecolor='orange', label='Dog')
]
plt.legend(handles=legend_elements, loc='upper right', fontsize=10)


plt.tight_layout()
plt.show()

plt.savefig(root_save_main + '/' + str(timenow) + '/val_correct_class_breeds.png', bbox_inches='tight')



