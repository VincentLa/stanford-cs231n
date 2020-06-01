# 2020-05-10

# 2020-05-03
Amil: Even in train,

Leila: Minor concern though. Ensembling is going to be important. Fusion will be important, dictates what components should be. So I didn't do this is in depth, read through the textbook, one way is to combine through layers, from different models not entirely sure. On very right, concatenation, chop off softmax layer you get more features

Tom: The FER (Facial Expression Recognition)

Amil: What's the strategy for facial. Even if we cache it to disk we need to do something with it.

Leila: The FER model does it do real time with video? It could read, read video input, change emotion label. How would we utilize that to read in video. Would we find emoti W to find overall sentiment of that one face? My face over course of one second, whichever one is the most frequent on that frame, that emotion label for that face.

Amil: We haven't read about what other people have done. 

Amil: From the frame you can extract a lot of features, faces, number of edges. Same thing with audio file, can extract a bunch of features from audio files. How are these features changing over time?

There is also a time dependence of this. 

Amil: Can a pure CNN approach solve the problem

Leila: Can also do audio encoders that can reduce the dimensionality of the input. We can have an audio encoder, have a bottle neck, latent features, take that and don't do the decoding stuff, take second to last encoding layer, feature set, but smaller. Feed that into a classifier. That could limit the number of features. 

Tom: What are you trying to encode? The entire frame?

Leila: Yes, you can encode the entire frame? I've worked with one compares how each row changes for all of the images.

Yolo: General Object detection -- need to feed it into a sentiment model after

Scene Sentiment and then an object sentiment