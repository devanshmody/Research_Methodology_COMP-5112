# Emotion Recognition From Speech (V1.0)

The understanding of emotions from voice by a human brain are normal instincts of human beings, but automating the process of emotion recognition from speech without referring any language or linguistic information remains an uphill grind. In this research work based on the input speech, I am trying to predict one of the six types of emotions (sad, neutral, happy, fear, angry, disgust). The diagram given below explain how emotion recognition from speech works. First audio features are extracted from input speech then those features are passed to the emotion recognition model which predicts one of the six emotion for the given input speech.
 ![Working Of Emotion Recgnition From Speech](https://user-images.githubusercontent.com/13017779/127468882-130282fb-9424-4366-a656-00c040232940.png)

# Motivation 

Most of the smart devices or voice assistants or robots exsisting in the world are not smart enough to understand the emotions. They are just like command and follow devices they have no emotional intelligence. When people talking to each other based on the voice they understand situation and react to it, for instance if someone is angry then other person will try to clam him by conveying in soft tone, these kind of harmonic changes are not possible with smart devices or voice assistants as they lack emtional intelligence. So adding emotions and making devices understand emotions will take them one step further to human like intelligence.

# Application

There are tonnes of applicates based on one can imagine. Few applications based on my thinking are human computer interaction using voice, Hhome automation,  anger/stress management by decoding emotions from voice, emotion recognition can help in detecting fear and cops can used this system to check if dialer is feared by some one or its just a normal call to register a complain, Marketing companies can use emotions to sell products based on user mood, autonomus vehicles can detect user emotion and adjust the speed of vehicles, It can help in solving psychological or depression problems. These are few applications according to me but there can be many more as voice based systems are increasing, even voice bsed chatting is common on social media platforms like clubhouse, discord, twitch, and others.

# Libraries and coding language used for the project
![languages](https://img.shields.io/github/languages/count/devanshmody/Research_Methodology_COMP-5112)
<a href="http://ffmpeg.org/"><img src="https://img.shields.io/badge/ffmpeg-green?style=flat&logo=ffmpeg&labelColor=green"></a>
<a href="https://pandas.pydata.org/"><img src="https://img.shields.io/badge/pandas-darkblue?style=flat&logo=pandas&labelColor=darkblue"></a>
<a href="https://numpy.org/"><img src="https://img.shields.io/badge/numpy-skyblue?style=flat&logo=numpy&labelColor=skyblue"></a>
<a href="https://www.tensorflow.org/"><img src="https://img.shields.io/badge/tensorflow-orange?style=flat&logo=tensorflow&labelColor=orange"></a>
<a href="https://docs.python.org/3/library/os.html"><img src="https://img.shields.io/badge/os-lightyellow?style=flat&logo=os&labelColor=lightyellow"></a>
<a href="https://docs.python.org/3/library/time.html"><img src="https://img.shields.io/badge/time-lightgreen?style=flat&logo=time&labelColor=lightgreen"></a>
<a href="https://librosa.org/"><img src="https://img.shields.io/badge/librosa-pink?style=flat&logo=librosa&labelColor=pink"></a>
<a href="https://docs.python.org/3/library/warnings.html"><img src="https://img.shields.io/badge/warnings-lightred?style=flat&logo=warings&labelColor=lightred"></a>
<a href="https://docs.python.org/3/library/base64.html"><img src="https://img.shields.io/badge/base64-lightgrey?style=flat&logo=base64&labelColor=lightgrey"></a>
<a href="https://pypi.org/project/google-colab/"><img src="https://img.shields.io/badge/google-colab-lightorange?style=flat&logo=google-colab&labelColor=lightorange"></a>
<a href="https://docs.python.org/3/library/glob.html"><img src="https://img.shields.io/badge/glob-lightgrey?style=flat&logo=glob&labelColor=lightgrey"></a>
<a href="https://docs.python.org/3/library/re.html"><img src="https://img.shields.io/badge/regex-darkgreen?style=flat&logo=regex&labelColor=darkgreen"></a>
<a href="https://scikit-learn.org/stable/"><img src="https://img.shields.io/badge/scikit-learn-darkorange?style=flat&logo=scikit-learn&labelColor=darkorange"></a>
<a href="https://keras.io/"><img src="https://img.shields.io/badge/keras-darkred?style=flat&logo=keras&labelColor=darkred"></a>
<a href="https://www.scipy.org/"><img src="https://img.shields.io/badge/scipy-violet?style=flat&logo=scipy&labelColor=violet"></a>
<a href="https://docs.python.org/3/library/io.html"><img src="https://img.shields.io/badge/io-grey?style=flat&logo=io&labelColor=grey"></a>
<a href="https://ipython.org/"><img src="https://img.shields.io/badge/ipython-purple?style=flat&logo=ipython&labelColor=purple"></a>
<a href="https://matplotlib.org/"><img src="https://img.shields.io/badge/matplotlib-brown?style=flat&logo=matplotlib&labelColor=brown"></a>
<a href="https://www.python.org/doc/"><img src="https://img.shields.io/badge/python3-yellow?style=flat&logo=python3&labelColor=yellow"></a>
![programming style](https://img.shields.io/badge/programming%20style-functional-brightgreen)
![programming language](https://img.shields.io/badge/programming%20language-python-red)

# Dataset description

I have used four datasets and all four datasets are freely available to downloaded from kaggle website. So I have downloaded the data, extracted and stored in my google drive.<br>
1) Ryerson Audio Visual Database of Emotional Speech and Song (Ravdess) dataset description:
   dataset link to download: "https://www.kaggle.com/uwrfkaggler/ravdess-emotional-speech-audio" <br>
   dataset stored on google drive at path: "/content/drive/MyDrive/Audiofiles/audio_speech_actors_01-24/"<br>
   dataset contains sub folders and file names as example in numbers format 03-01-01-01-01-01-01.wav.<br>
   Actor (01 to 24. Odd numbered actors are male, even numbered actors are female)<br>
   So based on the number there is a identifier for each number and its meaning as follows:<br>
   * Item1 Modality (01 = full-AV, 02 = video-only, 03 = audio-only).
   * Item2 Vocal channel (01 = speech, 02 = song).
   * Item3 Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
   * Item4 Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the 'neutral' emotion.
   * Item5 Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").
   * Item Repetition (01 = 1st repetition, 02 = 2nd repetition).<br>
   Therefore file 03-01-01-01-01-01-01.wav can be deduced as 03=audio-only,01=speech,01=neutral,01=normal,01=statement kids and 01=1st repetition.<br>
   
2) Crowd sourced Emotional Mutimodal Actors Dataset (CREMA-D) dataset description:
   dataset link to download: "https://www.kaggle.com/ejlok1/cremad" <br>
   dataset stored on google drive at path: "/content/drive/MyDrive/Audiofiles/AudioWAV/"<br>
   The format of files is 1001_DFA_ANG_XX.wav, where ANG stands for angry emotion.<br> 
   Similarly different emotion mappings are:<br>
   {'SAD':'sad','ANG':'angry','DIS':'disgust','FEA':'fear','HAP':'happy','NEU':'neutral'}
   
3) Toronto emotional speech set (Tess) dataset description:
   dataset link to download: "https://www.kaggle.com/ejlok1/toronto-emotional-speech-set-tess" <br>
   dataset stored on google drive at path: "/content/drive/MyDrive/Audiofiles/TESS Toronto emotional speech set data/"<br>
   There are folders in format OAF_angry, OAF_neural, OAF_disgust, YAF_sad and so on, where name after the underscore of the folder name contains the emotion        information, so the name after the underscore of the folder name is taken and files residing insider the folders are labeled accordingly.

4) Surrey Audio Visual Expressed Emotion (Savee) dataset description:
   dataset link to download: "https://www.kaggle.com/ejlok1/surrey-audiovisual-expressed-emotion-savee" <br>
   dataset stored on google drive at path: "/content/drive/MyDrive/Audiofiles/ALL/"<br>
   The files are in format DC_a01.wav where character 'a' contains the emotion information<br>
   Similarly different emotion mappings are:<br>
   {'a':'anger','d':'disgust','f':'fear','h':'happiness','n':'neutral','sa':'sadness','su':'surprise'}

# Universal decorator fucntion to calculate total time
```
def calc_time(func):
  def inner(*args, **kwargs):
    st = time.time()
    result = func(*args,**kwargs)
    end = time.time()-st
    print("Total time required: {:.3f} ms".format(end * 1000))
    return result
  return inner
```

# Description of important functions present in code (Model design and evaluation):

There are many functions in the program as functional programming style is used. Here I am going to describe few important functions which call other functions inside the functions and generate files and results. Detailed description of each function and its use can be found in code.
* Item1 Audio_features_extract() this function is used to extract audio features and generates a csv file at path "/content/drive/MyDrive/Audiofiles         /Audio_features_All_pr.csv" which contains audio features and their respective label information.
* Item2 Below given image shows snapshot of the csv file the file has total of 33954 rows × 179 columns.
  ![csv file snapshot](https://user-images.githubusercontent.com/13017779/127515316-3c4e2752-e376-4e71-ad76-513cec61bf1d.png)
* Item3 The csv file is loaded using pandas additional_preprocess() function carries out Exploratory Data Analysis and drop emotions with limited samples to avoid   missclassifications and then dataset is divided into train, test and validation set.
* Item4 Below image gives the detailed description of the whole process.
  ![Explorator Data Analysis and data preprocessing](https://user-images.githubusercontent.com/13017779/127515420-232f3180-34df-4531-8e34-93225748a0a6.png)
* Item5 Deep learning model for speech recognition is trained using the training data and at every epoch or checkpoint validation accuracy is calucated. The epoch   or checkpoint which gives highest validation accuracy the model is saved for that epoch or checkpoint at path " /content/drive/ MyDrive/Audiofiles/       
  emotionrecognition.hdf5", the model giving highest validation accuracy is only saved.
  ![model training snap shot](https://user-images.githubusercontent.com/13017779/127520834-e0b9fb86-2a60-4eed-a089-f28f5a028a48.png)

# Description of testing model in real time:
* The data for real time model testing is recorded using the microphone.
* The code to record audio speech using microphone is integrated from this link "https://ricardodeazambuja.com/deep_learning/2019/03/09/audio_and_video_google_     colab/".
* Then features are extracted from speech and passed to emotion recognition model which predicts one of the six emotions.
* Below figure shows the audio waveform and output of the emotion recognition model. 
 ![realtimeresult](https://user-images.githubusercontent.com/13017779/127523138-12df54f8-6af3-4907-9e80-56354bba12b8.png)

# Results 
* Below figure shows the training, testing and validation accuracy achieved by the emotion recognition model
  ![accuracy](https://user-images.githubusercontent.com/13017779/127524338-0209ab4e-eb82-4244-b519-e25cb4838859.png)
* Below figure shows the classification report and it can be seen in the report that for all the classes the value is greater than 0.5 which means the model can   
  predict the emotions accuratly to some extent. If the value is equal to 1.0 for all clases then it means model can predict accurrately always given the input   
  speech. But its diffcult to achieve real time prediction and 100% accuracy on real time envoiurment as there is noise and many other factors which can affect   
  the output. Given a challenge it can overcomed by training with big set of data in different languages to develop a universal model.
  ![classification](https://user-images.githubusercontent.com/13017779/127525847-6d2816a7-2e8b-4a3a-8385-e9c7a63bb870.png)
* The 0,1,2,3,4,5 in classification report resembles to different emotions which can be decoded from below image.
  ![emotionsmapping](https://user-images.githubusercontent.com/13017779/127526209-2d8748ca-2d99-4f70-ae11-1da5371cce61.png)
* below figure shows output of confusion matrix.<br>
  ![confumatrix](https://user-images.githubusercontent.com/13017779/127526415-1aca3e8f-32f7-44ac-bf34-fea0fd412209.png)
* below figure shows the training loss and accuracy curves despite the model giving and training accuracy of 100%, training and testing accuracy is near to    
  75%-76% but compared to the research work my model gives highest accuracy compared to those of the authors who carried out research work previously in this  
  area.
  ![curves](https://user-images.githubusercontent.com/13017779/127526942-9432d473-e6cc-4ef6-9a77-958ea56f3af0.png)
* Additionally to check wheather the model can on all types of voices and on unlabeled data a test was carried out using combination of different voices and 
  unlabled data. Below figure shows the results.
  ![unlabeltest](https://user-images.githubusercontent.com/13017779/127530261-ba33d4ea-640e-45ff-8bc9-7015eceb5e9f.png)
* Below figures shows comparison of my model with other authors who worked previously in this area of emotion recognition from speech
  ![comparison](https://user-images.githubusercontent.com/13017779/127533006-fac626bf-8bda-4bac-bbbb-fb72ef291f0a.png)
  
# Installation 

To download and run my google colab file 1130532_ResearchMethodology_Project_Final.ipynb following changes need to be made:
* Changes paths and give appropriate paths in the functions ravdess_data(), crema_data(), tess_data(), saveee_data(), fetch_data(), Audio_features_extract(),     
  audio_features_final(), emotion_recognition_model(), test_realtime(), evaluate_model(), unknown_audio() and diff_lang_test()
* If one dosent want to train the model just test the model then they can use the model file "emotion-recognition.hdf5", change paths in test_realtime() function   and they can test the model. 
* If you want to develop or implement or setupt at the whole code then as mentioned give proper paths and run all the functions its done.
* The main() function does all the work of training the model and evaluating the model. Once the main function completes running the model is file is generated     and can used for real time testing.

# usage 
* Once the model is trained and model file is generated one can use the test_realtime() function, this function invokes the microphone of the user records the       speech of the user, extracts the audio features and passes it to the emotion recognition model, then model predicts the emotions and displays it to the use.
* Before calling realtime_tested() function we call this function audio_features_final() which gives seven outputs out of which encoder output is important as it   contains the label (emotion) to number mapping.
* Functions to be called to test the model in real time enviourment
  ```
  x_train, x_test, y_train, y_test, x_val, y_val, encoder = audio_features_final()
  test_realtime(encoder)
  ```
* Also if using google colab make sure the function which are called inside the audio_features_final() and realtime_tested() are executed in advance as these two   functions are dependent on them.

# Support / Contact details
Given below are few of my social media accounts where anyone can contact me.<br>
<a href="https://in.linkedin.com/in/devansh-mody-5013aaab"><img src="https://img.shields.io/badge/LinkedIn-blue?style=flat&logo=linkedin&labelColor=blue"></a>
<a href="https://mobile.twitter.com/modydevansh"><img src="https://img.shields.io/badge/twitter-blue?style=flat&logo=twitter&labelColor=blue"></a>
<a href="https://www.youtube.com/channel/UCtc_46TMSXPUMpzVP0IAJUw"><img src="https://img.shields.io/badge/youtube-red?style=flat&logo=youtube&labelColor=red"></a>
<a href="https://www.instagram.com/devansh_mody/?hl=en"><img src="https://img.shields.io/badge/instagram-purple?style=flat&logo=instagram&labelColor=pink"></a>
<a href="https://devanshmody.blogspot.com/"><img src="https://img.shields.io/badge/My bloging website-yellow?style=flat&logo=blog&labelColor=lightyellow"></a>
<br>one can also contact me by mailing on my gmail id devanshmody2017@gmail.com<br>
For access to my google drive to see the setup of the whole project mail me on gmail idmentioned above access will be given to the selected people for some amount of time.

# Road-map (future ideas)

The backgorund noise may cause errors when testing the model in real time enviourment and thus it can affect the output of the model. To avoid the noise audio segmentation needs to be performed, so I am planning to develop an audio segmentation model which can seprate user speech from background noise so emotions can be predicted accurately. Also I will be collecting audio in different formats extract features and train the model so a universal model can be developed. Once audio model is build it can be applied to video also by combining audio model of emotion recognition with facial  model for emotion recognition this can help in acheving more accurate output. Additionally three models can be combined that is textual, voice and facial based but it requires huge computation power but can give a better output. Moreover I would like to build a audionet kind of embeddings similar to imagenet and word embeddings which will help other researchers working in this area to use pretrained audio embeddings.

# How to contribute

One can contribute by extracting features from different auido files the code for extracting features can be used from my ipynb file, different dataset may reqire different data preprocessing so one also write a function for data preprocessing and send me both prerporcessing code and csv file, so I can integrate both data preprocessing function and csv file with my csv file Audio_features_All_pr.csv. Additionally I am planning to build three model audio segmentation model, facial emotion recognition model and textual model so one can contribute by writing the function for the same and integrate it. Send me a git merge request to integrate code or contact me so we can check the integrity of code and combine the code.

# Authors / Acknowledgements

[1] Francesc Alı́as, Joan Claudi Socoró and Xavier Sevillano, ”A Review of Physical and Perceptual Feature Extraction Techniques for Speech, Music and Environmental Sounds”, Appl. Sci. 2016.[2] Kannan Venkataramanan and Haresh Rengaraj Rajamohan, ”Emotion Recognition from Speech”, arXiv:1912.10458v1 [cs.SD] 22 Dec 2019.<br>
[3] Haiyang Xu, Hui Zhang, Kun Han, Yun Wang, Yiping Peng and Xian-gang Li, ”Learning Alignment for Multimodal Emotion Recognition from Speech”, arXiv:1909.05645v2 [cs.CL] 3 Apr 2020.<br>
[4] Aharon Satt, Shai Rozenberg and Ron Hoory, ”Efficient Emotion Recognition from Speech Using Deep Learning on Spectrograms”, INTERSPEECH 2017, Stockholm, Sweden, August 20–24, 2017.<br>
[5] Jia Rong, Gang Li and Yi Ping Phoebe Chen, ”Acoustic feature selection for automatic emotion recognition from speech”, Information Processing and Management 45 (2009) 315–328.<br>
[6] https://ricardodeazambuja.com/deep_learning/2019/03/09/audio_and_video_google_colab/<br>
[7] https://keras.io/examples/audio/speaker_recognition_using_cnn/<br>
[8] https://librosa.org/doc/main/feature.html<br>

# License 

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

# Project Status

Emotion recognition model is ready and can be used in real time. The ipynb file can be downloaded and used by providing necesarry path changes. I am looking forward to develop other models mentioned in road-map (future ideas) and integrate all those models with my current emotion recognition model.









