import argparse
import numpy as np
import os.path
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.models import load_model
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from Keras.kaggle_utils import print_cm
from Keras.kaggle_model import cnn_lstm
from Keras.kaggle_jigsaw_word_embedding import KaggleJigsawWordEmbedding
from Keras.kaggle_jigsaw_loader import KaggleJigsawLoader

if __name__ == "__main__":

    data_dir = "~/.kaggle/competitions/jigsaw-toxic-comment-classification-challenge"

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-a", "--action", choices=["train", "test"], default="train",
                        help="set the action, either training or test the dataset")
    parser.add_argument("--train_dataset", default=os.path.join(data_dir, "train.csv"),
                        help="set the path to the training dataset")
    parser.add_argument("--test_dataset", default=os.path.join(data_dir, "test.csv"),
                        help="set the path to the test dataset")
    parser.add_argument("-m", "--model", default=None,
                        help="set the path to the pre-trained model/weights")
    parser.add_argument("--cv", type=bool, default=False,
                        help="enable / disable a full cross validation with n_splits=10")
    parser.add_argument("-b", "--batch_size", type=int, default=256,
                        help="set the batch size)")
    parser.add_argument("-e", "--epochs", type=int, default=100,
                        help="set the epochs number)")
    parser.add_argument("-d", "--dropout", type=float, default=0.1,
                        help="set the dropout)")
    parser.add_argument("-o", "--output", default=None,
                        help="set the path to the prediction output")
    parser.add_argument("-v", "--verbose", type=int, default=0,
                        help="set the verbosity)")

    args = parser.parse_args()

    # fix random seed for reproducibility
    seed = 70
    np.random.seed(seed)

    epochs = args.epochs
    train_dataset = args.train_dataset
    test_dataset = args.test_dataset

    dropout = args.dropout
    batch_size = args.batch_size
    validation_split = 0.1
    if args.model is None:
        model_file_path = "results/kaggle_jigsaw_d_{}.hdf5".format(dropout)
    else:
        model_file_path = args.model
    if args.output is None:
        output_file_path = "results/kaggle_jigsaw_prediction.csv"
    else:
        output_file_path = args.output

    model = cnn_lstm
    max_seq_length = 200
    # load train dataset
    pd = KaggleJigsawLoader(filename=train_dataset, validation_split=validation_split,
                            batch_size=batch_size, )
    docs = pd.get_docs()[:,0]
    word_embedding = KaggleJigsawWordEmbedding(docs=docs)
    embedding_matrix = word_embedding.get_embedding_matrix()
    vocab_size = word_embedding.get_vocabulary_size()
    pd.set_encoded_function(word_embedding.get_encoded_docs, max_seq_length=max_seq_length)

    if args.action == "train":
        tensorboard = TensorBoard(log_dir='graph', histogram_freq=0, write_graph=True, write_images=True)
        checkpoint = ModelCheckpoint(model_file_path, monitor='acc', verbose=args.verbose,
                                 save_best_only=True, mode='max')
        if args.cv:
            train_x, train_y = pd.get_dataset()
            kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
            estimator = KerasClassifier(build_fn=model, dropout=dropout,
                                        epochs=epochs, batch_size=500, verbose=args.verbose)
            results = cross_val_score(estimator, train_x, train_y, cv=kfold,
                                      fit_params={'callbacks':[checkpoint, tensorboard]})

            print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
        else:
            model = model(dropout=dropout, embedding_matrix=embedding_matrix, max_seq_length=max_seq_length)
            print(model.summary())
            # class_weight = {0:1, 1:1, 2:1, 3:1}
            history = model.fit_generator(generator = pd.generate("train"),
                                steps_per_epoch = pd.get_len("train")//batch_size + 1,
                                validation_data = pd.generate("validation"),
                                validation_steps = pd.get_len("validation")//batch_size + 1,
                                use_multiprocessing=True, class_weight=None,
                                epochs=epochs, verbose=args.verbose, callbacks=[checkpoint, tensorboard])
            print("Max of acc: {}, val_acc: {}".
                  format(max(history.history["acc"]), max(history.history["val_acc"])))
            print("Min of loss: {}, val_loss: {}".
                  format(min(history.history["loss"]), min(history.history["val_loss"])))
    else:
        # load test dataset
        pd = KaggleJigsawLoader(filename=test_dataset, batch_size=batch_size, validation_split=0.0, shuffle=False)
        pd.set_encoded_function(word_embedding.get_encoded_docs, max_seq_length=max_seq_length)

        # load model & weight
        loaded_model = load_model(model_file_path)
        print("Loaded model from disk")
        # evaluate loaded model on test data
        loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        predictions = []
        steps_per_epoch = pd.get_len("train")//batch_size + 1
        with open(output_file_path, "w") as prediction_output:
            prediction_output.write("id,toxic,severe_toxic,obscene,threat,insult,identity_hate\n")
            for (test_x, test_y, ids) in pd.generate("train", with_y=False, with_ids=True):
                if test_y is not None:
                    score = loaded_model.evaluate(test_x, test_y, verbose=0)
                    print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
                prediction = loaded_model.predict(test_x, verbose=0)
                predictions.extend(prediction)
                for i, id in enumerate(ids):
                    prediction_output.write("{},{}\n".
                                            format(id, ",".join(["{:.1f}".format(val) for val in prediction[i]])))

                steps_per_epoch -= 1
                if steps_per_epoch == 0:
                    break
            print("prediction: {}".format(len(predictions)))