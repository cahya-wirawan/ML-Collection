import pandas as pd
import numpy as np


class KaggleJigsawLoader(object):
    """
    KaggleJigsawLoader
    """
    x_indices = ["comment_text"]
    y_indices = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

    def __init__(self, filename, random_state=1, dim_x=len(x_indices), batch_size=32, shuffle=True,
                 validation_split=0.1):
        """
        :param filename:
        :param random_state:
        """
        self.dataset = {}
        self.dim_x = dim_x
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_state = random_state
        self.df = pd.read_csv(filepath_or_buffer=filename)
        self.encoded_docs_function = None

        self.ids = self.df["id"].values

        np.random.seed(self.random_state)
        # np.random.shuffle(self.ids)
        training_number = int(len(self.ids)*(1.0-validation_split))
        self.ids_train = self.ids[0:training_number]
        self.ids_validation = self.ids[training_number:]
        print(len(self.ids_train))
        print(len(self.ids_validation))

    def get_docs(self):
        docs = self.df[KaggleJigsawLoader.x_indices].values
        return docs

    def generate(self, type="train", with_ids=False):
        'Generates batches of samples'
        # Infinite loop
        while 1:
            # Generate order of exploration of dataset
            if type == "train":
                indexes = self.__get_exploration_order(self.ids_train)
            else:
                indexes = self.__get_exploration_order(self.ids_validation)
            # Generate batches
            imax = max(int(len(indexes)/self.batch_size), 1) + 1
            for i in range(imax):
                # Find list of IDs
                if type == "train":
                    ids_temp = [self.ids_train[k] for k in indexes[i*self.batch_size:(i+1)*self.batch_size]]
                else:
                    ids_temp = [self.ids_validation[k] for k in indexes[i*self.batch_size:(i+1)*self.batch_size]]
                ids_temp = np.array(ids_temp)

                # Generate data
                X, Y  = self.__data_generation(ids_temp)
                #X = np.expand_dims(X, axis=1)
                #y = np.expand_dims(Y, axis=1)
                if with_ids:
                    yield X, Y, ids_temp
                else:
                    yield X, Y

    def __get_exploration_order(self, ids):
        'Generates order of exploration'
        # Find exploration order
        indexes = np.arange(len(ids))
        if self.shuffle == True:
            np.random.shuffle(indexes)

        return indexes

    def __data_generation(self, ids):
        'Generates data of batch_size samples' # X : (n_samples, v_size, v_size, v_size, n_channels)
        # Initialization
        # X = np.empty((self.batch_size, self.dim_x), dtype=str)
        # Y = np.empty((self.batch_size))

        # Generate data
        X = self.df[(self.df['id'].isin(ids))][KaggleJigsawLoader.x_indices].values[:,0]
        Y = self.df[(self.df['id'].isin(ids))][KaggleJigsawLoader.y_indices].values
        if self.encoded_docs_function is not None:
            X = self.encoded_docs_function(X, self.max_seq_length)
        return X, Y

    def get_len(self, type="train"):
        if type == "train":
            return len(self.ids_train)
        else:
            if type == "validation":
                return len(self.ids_validation)
            else:
                return 0

    def sparsify(self, y, n_classes=4):
        'Returns labels in binary NumPy array'
        return np.array([[1 if y[i] == j else 0 for j in range(n_classes)]
                         for i in range(len(y))])

    def set_encoded_function(self, function, max_seq_length=100):
        self.encoded_docs_function = function
        self.max_seq_length = max_seq_length

    def get_dataset(self):
        """
        :return:
        """

        dataset_x = self.df[KaggleJigsawLoader.x_indices].values[:,0]
        if self.encoded_docs_function is not None:
            dataset_x = self.encoded_docs_function(dataset_x, self.max_seq_length)
        try:
            dataset_y = self.df[KaggleJigsawLoader.y_indices].values
        except KeyError:
            dataset_y = None
            # randomize the order of the datasets
        np.random.seed(self.random_state)
        #np.random.shuffle(dataset_x)
        if dataset_y is not None:
            np.random.seed(self.random_state)
            #np.random.shuffle(dataset_y)

        # dataset_x = np.expand_dims(dataset_x, axis=1)
        # dataset_y = np.expand_dims(dataset_y, axis=1)

        return dataset_x, dataset_y


if __name__ == "__main__":
    data_dir = "~/.kaggle/competitions/jigsaw-toxic-comment-classification-challenge"
    jigsaw_dataset = KaggleJigsawLoader(filename=data_dir + "/train_tiny.csv", batch_size=10)
    ds = []
    x = jigsaw_dataset.generate
    counter = 0
    for i in x("train"):
        ds.append(i)
        if counter == 10:
            break
        counter += 1
    print(x)
    #assert len(dataset_x) == 41
    #assert len(dataset_y) == 41
    #print(len(dataset_x), len(dataset_y))