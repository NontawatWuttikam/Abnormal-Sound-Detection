
─█▀▀█ █▀▀▄ █▀▀▄ █▀▀█ █▀▀█ █▀▄▀█ █▀▀█ █── 　 ░█▀▀▀█ █▀▀█ █──█ █▀▀▄ █▀▀▄ 
░█▄▄█ █▀▀▄ █──█ █──█ █▄▄▀ █─▀─█ █▄▄█ █── 　 ─▀▀▀▄▄ █──█ █──█ █──█ █──█ 
░█─░█ ▀▀▀─ ▀──▀ ▀▀▀▀ ▀─▀▀ ▀───▀ ▀──▀ ▀▀▀ 　 ░█▄▄▄█ ▀▀▀▀ ─▀▀▀ ▀──▀ ▀▀▀─

▒█▀▀▄ █▀▀ ▀▀█▀▀ █▀▀ █▀▀ ▀▀█▀▀ ░▀░ █▀▀█ █▀▀▄ 
▒█░▒█ █▀▀ ░░█░░ █▀▀ █░░ ░░█░░ ▀█▀ █░░█ █░░█ 
▒█▄▄▀ ▀▀▀ ░░▀░░ ▀▀▀ ▀▀▀ ░░▀░░ ▀▀▀ ▀▀▀▀ ▀░░▀

8 Classes binary classifier CNN model implemented using TensorFlow and Keras.


# 1. Details

![image](https://user-images.githubusercontent.com/73744769/126275006-acce2ff2-a4e9-49b8-b37d-fa6fab988a71.png)

This project uses trained 8 binary classifier convolutional neural network which was implemented on DenseNet201 architecture. It receives audio input stream and stack them til it reaches 10 seconds long then convert the stacked audio to Mel Frequency Spectrogram and feed as an input for each binary model. The model will output the result as a probability value wether the audio contains the target noise or not. if the probability is higher than a fixed - threshold value, it will display the detection on the window.


# 2. Project setup & structure

![image](https://user-images.githubusercontent.com/73744769/126282616-a9d4f355-1363-4b33-86f8-20facaed9f40.png)

To run the project, please install Python 3.7.7 on your machine (using VirtualEnv is recommended). Then, create a folder for a project and put audio_binary.py file, requirements.txt file, and binary_model folder into the project folder. You can see the example of project structure above. Then install the essential modules in requirements.txt.

DOWNLOAD binary_model HERE => https://drive.google.com/drive/folders/1B6Y8jbxgfUNPEb7prjxcYhR37WYzSJe_?usp=sharing


# 3. Datasets Repository

We use audio dataset provided by Google AudioSet. The repository provide diverge audio for every type of sound in real life by marking the target sound in YouTube videos and gather them into one large spreadsheet. The spreadsheet contains YouTube video url, start and end range, and class label index. You can extract the audio clip from the spreadsheet filtered by your desired classes easily using CSV_Filter_Unbalanced.ipynb and Youtube_Downloader.ipynb. For more detail, please check out AudioSet Website. http://research.google.com/audioset/index.html


# 4. Dataset Preprocessing

After we extracted the audio in each class and separated them into sub-foders. Here is the structure of the dataset folder.

![image](https://user-images.githubusercontent.com/73744769/126277438-fb29047b-7a16-49fa-9bb2-c238f213a958.png)

We preprocess the audio inside the folder by converting them into Mel Frequency Spectrogram using Librosa with the parameters in the .ipynb. Then we normalize, reshape and change the color chanel into RGB chanels for the spectrogram to make them suitable for DenseNet201 architecture.

![image](https://user-images.githubusercontent.com/73744769/126278040-df92db9d-1973-4ebd-8666-08b693e385f6.png)


# 5. Training 

We train the model on Google Colab instance (free). Each model is created using Keras Application Module that contains pre - implemented graph for DenseNet201. The initial weight for each model is random. We train the model using 22,729 training data and 1393 testing data. the model was trained with approximantely 10 epoches and 16 batch size.


# 6. Evaluation

We evaluate trained models using Precision, Recall, F1 score, ROC curve, and confusion matrix. The evaluation for each binary model is recorded HERE => https://drive.google.com/drive/folders/1GQBVhZrFNWNkxHJ0RdDGD8qAHekhfmMk?usp=sharing

| model          | trained epoch | positive precision | positive recall | positive F1 |
|----------------|---------------|--------------------|-----------------|-------------|
| Barking        |       9       |        0.465       |      0.920      |    0.617    |
| Car Alarm      |       12      |        0.15        |      0.633      |    0.242    |
| Crying         |       18      |        0.608       |      0.662      |    0.634    |
| Explosion      |       11      |        0.403       |      0.815      |     0.54    |
| Interior Alarm |       17      |        0.755       |      0.703      |    0.728    |
| Gunshot        |       16      |        0.42        |       0.93      |     0.58    |
| Screaming      |       8       |        0.168       |      0.643      |    0.266    |
| Siren          |       8       |        0.76        |      0.672      |    0.713    |


