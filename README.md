# stanford-cs231n
Stanford CS 231n Convolutional Neural Networks

Course website: [http://cs231n.stanford.edu/](http://cs231n.stanford.edu/)

# Google Cloud
Go to [Compute Engine](https://console.cloud.google.com/compute/instances?project=stanford-cs-231n&authuser=1&organizationId=733583539713&instancessize=50) on console.cloud.google.com

Note, the credentials for this project are:

Acount: Law Stanford
Project Organization: law.stanford.edu

Click on the stanford-cs-231n-vm VM instance, and click "Start". Once you turn on the engine, type:

```
gcloud beta compute --project "stanford-cs-231n" ssh --zone "us-west1-b" "stanford-cs-231n-vm"
```

# Jupyter Notebook
Note, that once remote, to access the Jupyter Notebook, type

```
jupyter notebook
```

in terminal as you normally would to start the jupyter notebook server. However, on the browser, go to:

```
http://35.212.157.230:8888/
```

This is based off the static IP in Google Cloud

## To Push back to GitHub
Note, that to push back to GitHub, with 2FA, you get an authentication issue. For now, what I did, was generated a token (https://medium.com/@ginnyfahs/github-error-authentication-failed-from-command-line-3a545bfd0ca8). The actual token is stored as an environment variable GITHUB_TOKEN (in my local bash_profile)

# Setup Instructions
For instructions on setting up Google Cloud: https://github.com/cs231n/gcloud

Setup instructions for Google Colab: https://cs231n.github.io/setup-instructions/#working-remotely-on-google-colaboratory (or can basically just google around)

Remember, if you are using Sublime Text 3 locally, you have to push changes to git and pull inside the VM to ensure that changes get passed through. Because of this, you may want to set up two terminal windows, one for Jupyter notebook and one to pull.

# Lecture Notes

## K Nearest Neighbor
http://cs231n.github.io/classification/

# Other Public Class Resources
1. https://github.com/jariasf/CS231n
2. https://github.com/srinadhu/CS231n
3. https://github.com/Arnav0400/CS231n-2019

# Recording Lectures
We can record lectures by using Quicktime Player (which comes for free on Mac). Click Record Entire Screen. Also make sure to click on Microphone to record audio.

# Other Helpful Notes
1. http://cs229.stanford.edu/section/cs229-linalg.pdf