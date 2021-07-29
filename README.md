# Emotion Recognition From Speech

The understanding of emotions from voice in a human brain are normal instincts of human beings, but automating the process of emotion recognition from speech without referring any language or linguistic information remains an uphill grind. In this research work based on the input speech, I am trying to predict one of the six types of emotions (sad, neutral, happy, fear, angry, disgust). The diagram given below explain how emotion recognition from speech works. First audio features are extracted from input speech then those features are passed to the emotion recognition model which predicts one of the six emotion for the given input speech.
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




<a href="https://in.linkedin.com/in/devansh-mody-5013aaab"><img src="https://img.shields.io/badge/LinkedIn-blue?style=flat&logo=linkedin&labelColor=blue"></a>
<a href="https://mobile.twitter.com/modydevansh"><img src="https://img.shields.io/badge/twitter-blue?style=flat&logo=twitter&labelColor=blue"></a>
<a href="https://mobile.twitter.com/modydevansh"><img src="https://img.shields.io/badge/twitter-blue?style=flat&logo=twitter&labelColor=blue"></a>
<a href="https://www.youtube.com/channel/UCtc_46TMSXPUMpzVP0IAJUw"><img src="https://img.shields.io/badge/youtube-red?style=flat&logo=youtube&labelColor=red"></a>
<a href="https://www.instagram.com/devansh_mody/?hl=en"><img src="https://img.shields.io/badge/instagram-purple?style=flat&logo=instagram&labelColor=pink"></a>
<a href="https://devanshmody.blogspot.com/"><img src="https://img.shields.io/badge/My bloging website-yellow?style=flat&logo=blog&labelColor=lightyellow"></a>










