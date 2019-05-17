import os
from argparse import Namespace
import torch.optim as optim
import torch
from surname_dataset import SurnameDataset
import surname_dataset as dataset_utils
from surname_classifier import SurnameClassifier
import torch.nn as nn
import json

args = Namespace(
    # Data and path information
    model_state_file='model.pth',
    surname_csv='./data/surnames_with_splits.csv',
    save_dir='./model_storage/surname/',
    vectorizer_file='vectorizer.json',
    hidden_size=300,
    # No model hyperparameters
    # Traning hyperparameters
    batch_size=64,
    early_stoping_criteria=5,
    learning_rate=0.001,
    num_epochs=20,
    seed=1337,
    # Runtime options omitted for space
)


def compute_accuracy(y_pred, y_target):
    y_target = y_target.cpu()
    y_pred_indices = torch.argmax(torch.softmax(y_pred, dim=1), dim=1).cpu().long()  # .max(dim=1)[1]
    n_correct = torch.eq(y_pred_indices, y_target).sum().item()
    return n_correct / len(y_pred_indices) * 100


def make_train_state(args):
    return {
        'epoch_index': 0,
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'test_loss': -1.,
        'test_acc': -1.
    }


train_state = make_train_state(args)

if not torch.cuda.is_available():
    args.cuda = False

args.device = torch.device("cuda" if args.cuda else "cpu")

# dataset and vectorizer
dataset = SurnameDataset.load_dataset_and_make_vectorizer(args.surname_csv)
vectorizer = dataset.get_vectorizer()

# model
classifier = SurnameClassifier(input_size=len(vectorizer.surname_vocab),
                               hidden_size=args.hidden_size,
                               output_size=len(vectorizer.nationality_vocab))
classifier = classifier.to(args.device)

# loss and optimizer
loss_func = nn.CrossEntropyLoss(weight=dataset.class_weights)
optimizer = optim.Adam(classifier.parameters(), lr=args.learning_rate)

# training loop
for epoch_index in range(args.num_epochs):
    train_state['epoch_index'] = epoch_index

    # Iterate over training dataset

    # setup: batch generator, set loss and acc to 0, set train mode on
    dataset.set_split('train')
    batch_generator = dataset_utils.generate_batches(dataset, batch_size=args.batch_size, device=args.device)
    running_loss = 0.0
    running_acc = 0.0
    classifier.train()

    for batch_index, batch_dict in enumerate(batch_generator):
        # the training routine is 5 steps:-----------------------

        # step 1: zero the gradients
        optimizer.zero_grad()

        # step 2: compute the output
        y_pred = classifier(x_in=batch_dict['x_data'].float())

        # step 3: compute the loss
        loss = loss_func(y_pred, batch_dict['y_target'])
        loss_batch = loss.item()
        running_loss += (loss_batch - running_loss) / (batch_index + 1)

        # step 4: use loss to produce gradients
        loss.backward()

        # step 5: use optimizer to take gradient step
        optimizer.step()

        # ----------------------------------------------------------
        # compute the accuracy
        acc_batch = compute_accuracy(y_pred, batch_dict['y_target'])
        running_acc += (acc_batch - running_acc) / (batch_index + 1)

    train_state['train_loss'].append(running_loss)
    train_state['train_acc'].append(running_acc)

    # Iterate over val dataset

    # setup: batch genrator, set loss and acc to 0, set eval mode on
    dataset.set_split('val')
    batch_generator = dataset_utils.generate_batches(dataset, batch_size=args.batch_size, device=args.device)
    running_loss = 0
    running_acc = 0
    classifier.eval()

    for batch_index, batch_dict in enumerate(batch_generator):
        # step 1: compute the output
        y_pred = classifier(x_in=batch_dict['x_data'].float(), train_mode=False)

        # step 2: compute the loss
        loss = loss_func(y_pred, batch_dict['y_target'])
        loss_batch = loss.item()
        running_loss += (loss_batch - running_loss) / (batch_index + 1)

        # step 3: compute the accuracy
        acc_batch = compute_accuracy(y_pred, batch_dict['y_target'])
        running_acc += (acc_batch - running_acc) / (batch_index + 1)
    print("Epoch: {}, val_loss: {:.3f}, val_acc: {:.2f}%".format(epoch_index, running_loss, running_acc))
    train_state['val_loss'].append(running_loss)
    train_state['val_acc'].append(running_acc)

# make dir if need and save model
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
torch.save(classifier.state_dict(), os.path.join(args.save_dir, args.model_state_file))
# save SurnameVectorizer
with open(os.path.join(args.save_dir, args.vectorizer_file), 'w') as outfile:
    json.dump(dataset.get_vectorizer().to_serializable(), outfile)

# test model
classifier.load_state_dict(torch.load(os.path.join(args.save_dir, args.model_state_file)))
dataset.set_split('test')
batch_generator = dataset_utils.generate_batches(dataset, batch_size=args.batch_size, device=args.device)
running_loss = 0.
running_acc = 0.
classifier.eval()

for batch_index, batch_dict in enumerate(batch_generator):
    # compute output
    y_pred = classifier(x_in=batch_dict['x_data'].float(), train_mode=False)

    # compute loss
    loss = loss_func(y_pred, batch_dict['y_target'])
    loss_batch = loss.item()
    running_loss += (loss_batch - running_loss) / (batch_index + 1)

    # compute accuracy
    acc_batch = compute_accuracy(y_pred, batch_dict['y_target'])
    running_acc += (acc_batch - running_acc) / (batch_index + 1)

train_state['test_loss'] = running_loss
train_state['test_acc'] = running_acc

print("Test loss {:.3f}".format(train_state['test_loss']))
print("Test acc {:.2f}%".format(train_state['test_acc']))
