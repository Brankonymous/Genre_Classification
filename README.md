# Genre_Classification
This project is a part of a research whose aim is to compare 4 different feature extraction methods for music genre classification. Compared methods are spectrum, MFCC, LPCC and fingerprint method. Research was conducted as a part of an Applied physics and electronics seminar in a research station "Petnica".
Research paper is still in progress, but the current version can be found under the name **"research.pdf"** (unfortunately, it is currently available only in Serbian).


Two different databases were used - GTZAN and NATA database. GTZAN is publicly available database, while NATA is a custom database that we used. 
As a result, it won't be possible to recreate the results that we got by using the source code provided here. That is why we won't publish all the files that we used, since it would be untidy and hard to read. We provide files that were used for feature extration, files that were used to declare, train and test neural networks, as well as files that generated confusion matrices. 

Research results, for GTZAN and NATA database respectively, are shown in the images below:

![Capture](https://user-images.githubusercontent.com/43354887/108545752-7cff0900-72e8-11eb-9af7-5d4bd7961bc3.JPG)

As you can see, the overall best result was achieved with the combination of MFCC and Spectrum (Spektar) features.
