import os
from argparse import Namespace
import torch
from surname_dataset import SurnameDataset, SurnameVectorizer
from surname_classifier import SurnameClassifier
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args = Namespace(
    # Data and path information
    model_state_file='model.pth',
    surname_csv='./data/surnames_with_splits.csv',
    save_dir='./model_storage/surname/',
    vectorizer_file='vectorizer.json',
    hidden_size=300,
    device=device,
    # No model hyperparameters
    # Traning hyperparameters
    batch_size=64,
    early_stoping_criteria=5,
    learning_rate=0.001,
    num_epochs=20,
    seed=1337,
    # Runtime options omitted for space
)


def preprocess_text(raw_text):
    """
    Clean text before vectorization
    :param raw_text: input before clean
    :return:
    """
    return raw_text


def predict_surname(text, classifier, vectorizer):
    """
    Predict the rating of a surname
    :param text: text of the surname
    :param classifier: trained model
    :param vectorizer: corresponding vectorizer
    :return:
    """

    surname = preprocess_text(text)
    vectorized_surname = torch.tensor(vectorizer.vectorize(surname))
    result = classifier(vectorized_surname.view(1, -1).to(args.device))
    probability_value = torch.softmax(result, dim=0).squeeze()

    class_index = torch.argmax(probability_value).item()

    return vectorizer.nationality_vocab.lookup_index(class_index)


if __name__ == "__main__":
    with open(os.path.join(args.save_dir, args.vectorizer_file)) as json_file:
        contents = json.load(json_file)
    # dataset and vectorizer
    vectorizer = SurnameVectorizer.from_serializable(contents)

    # model
    classifier = SurnameClassifier(input_size=len(vectorizer.surname_vocab),
                                   hidden_size=args.hidden_size,
                                   output_size=len(vectorizer.nationality_vocab))
    classifier = classifier.to(args.device)

    # load model
    classifier.load_state_dict(torch.load(os.path.join(args.save_dir, args.model_state_file), map_location=args.device))
    # get vectorizer to convert text

    test_surname = "nguyen"
    prediction = predict_surname(test_surname, classifier=classifier, vectorizer=vectorizer)
    print("{} -> {}".format(test_surname, prediction))
