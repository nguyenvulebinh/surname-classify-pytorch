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
    # Model hyper parameter
    char_embedding_size=100,
    rnn_hidden_size=64,
    # Traning hyperparameters
    num_epochs=100,
    learning_rate=1e-3,
    batch_size=64,
    seed=1337,
    early_stopping_criteria=5,
    # Runtime hyper parameter
    cuda=True,
    catch_keyboard_interrupt=True,
    reload_from_files=False,
    expand_filepaths_to_save_dir=True,
    device=device
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
    vectorized_surname, len_surname = vectorizer.vectorize(surname)
    result = classifier(torch.tensor(vectorized_surname).unsqueeze(dim=0).to(args.device), torch.tensor([len_surname]))
    probability_value = torch.softmax(result, dim=0).squeeze()

    class_index = torch.argmax(probability_value).item()

    return vectorizer.nationality_vocab.lookup_index(class_index)


if __name__ == "__main__":
    with open(os.path.join(args.save_dir, args.vectorizer_file)) as json_file:
        contents = json.load(json_file)
    # dataset and vectorizer
    vectorizer = SurnameVectorizer.from_serializable(contents)

    # model
    classifier = SurnameClassifier(embedding_size=args.char_embedding_size,
                                   vocab_size=len(vectorizer.surname_vocab),
                                   num_classes=len(vectorizer.nationality_vocab),
                                   rnn_hidden_size=args.rnn_hidden_size,
                                   padding_idx=vectorizer.surname_vocab.mask_index)
    classifier = classifier.to(args.device)

    # load model
    classifier.load_state_dict(torch.load(os.path.join(args.save_dir, args.model_state_file), map_location=args.device))
    # get vectorizer to convert text

    test_surname = "nguyen"
    prediction = predict_surname(test_surname, classifier=classifier, vectorizer=vectorizer)
    print("{} -> {}".format(test_surname, prediction))
