from django.shortcuts import render

from django.shortcuts import render


from django.shortcuts import render,redirect
from django.contrib.auth.models import User
from django.contrib import messages
from django.contrib.auth import login ,logout,authenticate
from django.core.files.storage import default_storage
import tempfile

# Create your views here.

def home(request):
    return render(request,'Home.html')

def register(request):
    if request.method == 'POST':
        First_Name = request.POST['name']
        Email = request.POST['email']
        username = request.POST['username']
        password = request.POST['password']
        confirmation_password = request.POST['cnfm_password']
        if password == confirmation_password:
            if User.objects.filter(username=username).exists():
                messages.error(request, 'Username already exists, please choose a different one.')
                return redirect('register')
            else:
                if User.objects.filter(email=Email).exists():
                    messages.error(request, 'Email already exists, please choose a different one.')
                    return redirect('register')
                else:
                    user = User.objects.create_user(
                        username=username,
                        password=password,
                        email=Email,
                        first_name=First_Name,
                    )
                    user.save()
                    return redirect('login')
        else:
            messages.error(request, 'Passwords do not match.')
        return render(request, 'register.html')
    return render(request, 'register.html')

def login_view(request):
    if request.method == "POST":
        username = request.POST.get('username')
        password = request.POST.get('password')
        if User.objects.filter(username=username).exists():
            user=User.objects.get(username=username)
            if user.check_password(password):
                user = authenticate(username=username,password=password)
                if user is not None:
                    login(request,user)
                    messages.success(request,'login successfull')
                    return redirect('/')
                else:
                   messages.error(request,'please check the Password Properly')
                   return redirect('login')
            else:
                messages.error(request,"please check the Password Properly")  
                return redirect('login') 
        else:
            messages.error(request,"username doesn't exist")
            return redirect('login')
    return render(request,'login.html')
# Load and preprocess the dataset
def logout_view(request):
    logout(request)
    return redirect('login')


# ML concepts
import matplotlib.pyplot as plt
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers.core import Dense,Activation,Dropout, Flatten
import seaborn as sns
import os
import cv2
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
import pickle
from keras.models import model_from_json
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')
path = r"static/augmented_images"
model_folder = "static/Model2"
#This will be used as the folder name where the model will be stored.
categories = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
categories
X_file = os.path.join(model_folder, "X.txt.npy") ## Construct file paths for loading and saving NumPy arrays
Y_file = os.path.join(model_folder, "Y.txt.npy")
if os.path.exists(X_file) and os.path.exists(Y_file): ## Check if the files alreadyexist
    X = np.load(X_file)
    Y = np.load(Y_file) ## Load the arrays from the files
    print("X and Y arrays loaded successfully.")
else:
    ## Initialize empty arrays for input and output
    X = [] # input array
    Y = [] # output array
    # Traverse through the directory specified by 'path'
    for root, dirs, directory in os.walk(path):
    # Loop through the files in the directory
        for j in range(len(directory)):
            name = os.path.basename(root)      #extract the category name from the directory path
            print(f'Loading category: {dirs}')     # Print the category being loaded
            print(name+" "+root+"/"+directory[j])    # Print the path of the current image being loaded
            if 'Thumbs.db' not in directory[j]:     ## Check if the current file is not 'Thumbs.db'
                img_array = cv2.imread(root+"/"+directory[j])   ## Read the image fileusing OpenCV
                img_resized = cv2.resize(img_array, (64,64))   # Resize the image to 64x64 pixels
                im2arr = np.array(img_resized)      # Convert the resized image to a NumPy array
                im2arr = im2arr.reshape(64,64,3)    # Reshape the array to match theexpected input shape (64x64x3)
                X.append(im2arr)       # Append the index of the category in categories list to Y
                Y.append(categories.index(name)) # Append the index of the category incategories list to the output array (Y)
    X = np.asarray(X)        ## Convert the lists to NumPy arrays
    Y = np.asarray(Y)
    X = X.astype('float32')      # Convert pixel values to float32 and normalize them to arange between 0 and 1
    X = X / 255               # Normalize pixel values
    Y = to_categorical(Y, num_classes=len(categories))         # Convert labels to one-hot encoding
## Save the processed arrays to files
    np.save(X_file, X)
    np.save(Y_file, Y)
    print("X and Y arrays saved successfully.")
# Shuffle the data using randomly generated indices
indices = np.arange(X.shape[0]) # it give shape of that matrix
np.random.shuffle(indices)
X = X[indices]
Y = Y[indices]
num_classes = len(categories)
num_classes

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.20,random_state=0)
labels = ['0', '1']  
accuracy, precision, recall, fscore = [], [], [], []  # Lists to store metrics

def calculateMetrics(algorithm, predict, testY, labels):
    testY = testY.astype('int')
    predict = predict.astype('int')

    # Calculate metrics
    a = accuracy_score(testY, predict) * 100
    p = precision_score(testY, predict, average='macro') * 100
    r = recall_score(testY, predict, average='macro') * 100
    f = f1_score(testY, predict, average='macro') * 100

    # Append metrics to lists
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)

    # Print metrics
    print(f"{algorithm} Accuracy: {a:.2f}")
    print(f"{algorithm} Precision: {p:.2f}")
    print(f"{algorithm} Recall: {r:.2f}")
    print(f"{algorithm} F1 Score: {f:.2f}")
    print(f"")
    print(f"")

    # Classification report
    report = classification_report(testY, predict, target_names=labels, output_dict=True)
    conf_matrix = confusion_matrix(testY, predict)
   # Compute and print per-class metrics
    print(f"{algorithm} Classification Report:")
    print(f"")

    class_accuracy = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
    print(f"{'Class':<10}{'Accuracy':<10}{'Precision':<10}{'Recall':<10}{'F1 Score':<10}")
    for i, class_name in enumerate(labels):
        print(f"{class_name:<10}{class_accuracy[i]:<10.4f}{report[class_name]['precision']:<10.4f}{report[class_name]['recall']:<10.4f}{report[class_name]['f1-score']:<10.4f}")
    
    # Confusion matrix
    plt.figure(figsize=(5, 5))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(f"{algorithm} Confusion Matrix")
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()
# Check if the pkl file exists
model =None
def custom_cnn(request):
    global model

    Model_file = os.path.join(model_folder, "DLmodel.json")
    Model_weights = os.path.join(model_folder, "DLmodel_weights.h5")
    Model_history = os.path.join(model_folder, "history.pckl")

    if os.path.exists(Model_file):
        # If the model JSON file exists, load the model
        with open(Model_file, "r") as json_file:
            loaded_model_json = json_file.read()
            model = model_from_json(loaded_model_json)
        
        # Load weights
        model.load_weights(Model_weights)

        # Print the model summary to verify it's loaded correctly
        print(model.summary())

        # Load the training history (if required)
        with open(Model_history, 'rb') as f:
            accuracy = pickle.load(f)

        acc = accuracy['accuracy'][9] * 100  # 10th epoch accuracy
        print(f"CNN Model Prediction Accuracy = {acc}")

    else:
        # If the model doesn't exist, create and train a new model
        model = Sequential()
        model.add(Convolution2D(32, 3, 3, input_shape=(64, 64, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Convolution2D(32, 3, 3, activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        print(model.summary())

        # Train the model
        hist = model.fit(X_train, Y_train, batch_size=16, epochs=50, validation_data=(X_test, Y_test), shuffle=True, verbose=2)

        # Save the trained model weights and structure
        model.save_weights(Model_weights)
        model_json = model.to_json()
        with open(Model_file, "w") as json_file:
            json_file.write(model_json)

        # Save training history
        with open(Model_history, 'wb') as f:
            pickle.dump(hist.history, f)

        # Extract accuracy from history
        accuracy = hist.history
        acc = accuracy['accuracy'][9] * 100

    return render(request, 'prediction.html', {
        'algorithm': 'Custom CNN',
        'accuracy': acc,
    })
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Example class labels, modify as necessary
def CNN_model(request):
    if model==None:
        messages.error(request,'please Train the custom CNN first')
        return redirect('/')
    else:
        import numpy as np
        global X_test, Y_test  # Make sure X_test, Y_test are declared as global if they are defined elsewhere

        # Check the shape of Y_test (you can remove this check if Y_test is one-hot encoded already)
        if Y_test.ndim > 1 and Y_test.shape[1] > 1:
            Y_test = np.argmax(Y_test, axis=1)  # Convert one-hot encoding to class labels if necessary

        # Your model prediction code
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)

        # Now the code should work without throwing an UnboundLocalError
        categories = ['0', '1']  # Adjust to match your actual class names
        calculateMetrics("Hybrid CNN Model", y_pred_classes, Y_test, categories)
        
        return render(request, 'prediction.html', {
            'algorithm': 'CNN Classifier',
            'accuracy': accuracy[-1],
            'precision': precision[-1],
            'recall': recall[-1],
            'fscore': fscore[-1]
        })

def prediction(request):
    Test=True
    if request.method=='POST' and request.FILES['file']:
        file=request.FILES['file']
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')  # Temporary file with .jpg extension
        temp_file_path = temp_file.name  # Get the path of the temporary file
        with open(temp_file_path, 'wb') as temp_image:
            for chunk in file.chunks():
                temp_image.write(chunk)
        img = cv2.imread(temp_file_path)
        img_resized = cv2.resize(img, (64,64))    # every image in different size hence we resize the image.
        im2arr = np.array(img_resized)           # Convert the resized image into a NumPy array
        img_reshaped = im2arr.reshape(1,64,64,3)         ### Reshape the NumPy array to match the expected input shape of the model (1, 64, 64, 3)

        test = np.asarray(img_reshaped)
        # Convert the reshaped image array into a NumPy array

        test = test.astype('float32')
        test = test/255
        pred_probability = model.predict(test)
        pred_number = np.argmax(pred_probability)   #argmax is a fundamental function in deeplearning for

        # converting probability distributions into class labels
        ## This returns the index of the maximum value in the pred_probability array,
        # which corresponds to the predicted class label.
        categories=['Healthy','UnHealthy']
        output_name=categories[pred_number] # Get the corresponding output class name basedon the index
        plt.imshow(img)
        plt.text(10, 10, f'Predicted Output: {output_name}', color='white', fontsize=12, weight='bold', backgroundcolor='black')
        plt.axis('off')
        plt.show()
        return render(request,'prediction.html',{'predict':categories[pred_number],'test':Test})
    return render(request,'prediction.html',{'test':Test})