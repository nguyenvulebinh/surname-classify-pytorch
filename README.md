# Surname Classification using PyTorch

#### The surnames dataset, a collection of 10,000 surnames from 18 different nationalities collected by the authors from different name sources on the internet. The top three classes account for more than 60% of the data: 27% are English, 21% are Russian, and 14% are Arabic. The remaining 15 nationalities have decreasing frequency.

#### This repo is baseline for that problem. I also provide pipeline to preprocess data before input to the model.

* surname_classifier.py define model (Multi perceptron layer in branch mlp, CNN layer in branch master. CNN is better)
* surname_dataset.py define classes for prepare data (convert from csv to vector, vocab, create batch, ...)
* train.py steps to train, save model.
* infer.py infer new instance

Happy coding!!!!!

