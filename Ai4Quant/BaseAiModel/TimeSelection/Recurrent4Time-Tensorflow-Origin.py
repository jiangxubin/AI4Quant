from utils import DataIO
import tensorflow as tf


# parser = argparse.ArgumentParser()
# parser.add_argument('batch_size', type=int, help="Number of examples of each Bacth", default=32)
# parser.add_argument('time_step', type=int, help="Length of time steps of Sequence model", default=10)
# parser.add_argument("step_vector_size", type=int, help="Number of features of each example", default=5)
# parser.add_argument("dropout_ratio", type=float, help="Ratio to random dropuout neurons", default=0.5)
# parser.add_argument("epochs", type=int, help="Num of how much model run through all models", default=50)
# args = parser.parse_args()

batch_size = 16
# batch_size = args.batch_size
time_step = 10
hidden_units_1 = 5
hidden_units_2 = 5
layer_num = 2
learning_rate = 0.0006
class_num = 2
# time_step = args.time_step
step_vector_size = 5
# step_vector_size = args.step_vector_size
dropout_ratio = 0.8
# dropout_ratio = args.dropout_ratio
epochs = 300
# epochs = args.epochs


class LSTM4Regression:
    def __init__(self, universe: list, start_date: str, end_date: str):
        """
        :param universe: 初始化股票池
        """
        self.universe = universe
        self.start_date = start_date
        self.end_date = end_date
        self.all_df = []
        self.model = None

    def get_train_data_fn(self)->tuple:
        """
        Get X for feature and y for label
        :return: DataFrame of raw data
        """
        raw_data = DataIO.StockRawData.get_universe_data(self.universe, start_date=self.start_date, end_date=self.end_date)# get raw data
        X, y = DataIO.FatureEngineering.feature_label_split(raw_data)
        return X, y

    def get_test_data_fn(self):
        """
        Build data pipline for test data
        :return: a dict of features and a tensor of labels
        """
        return None

    def __build_lstm_model(self, X):
        """
        Build LSTM model for train, validation and predict
        :return: LSTM model
        """
        # Add a lstm layer to create basic lstm_cell_1
        lstm_cell_1 = tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_units_1, name='lstm_1')
        # Create a dropout wrapper of lstm_cell_1
        lstm_cell_1 = tf.nn.rnn_cell.DropoutWrapper(cell=lstm_cell_1, input_keep_prob=1.0, output_keep_prob=dropout_ratio)
        # Add a lstm layer to create basic lstm_cell_2
        lstm_cell_2 = tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_units_2, name='lstm_2')
        # Create a dropout wrapper of lstm_cell_2
        lstm_cell_2 = tf.nn.rnn_cell.DropoutWrapper(cell=lstm_cell_2, input_keep_prob=1.0, output_keep_prob=dropout_ratio)
        # Create a multi layer lstm layers
        multi_layer_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_1, lstm_cell_2], state_is_tuple=True)
        # Initialize the hidden state
        ini_state = multi_layer_cell.zero_state(batch_size=batch_size, dtype=tf.float32)
        # Build the model
        output_hidden_states, last_states = tf.nn.dynamic_rnn(multi_layer_cell, inputs=X, initial_state=ini_state, time_major=False)
        # Get the hidden state of last time step
        h_state = tf.nn.softmax(output_hidden_states[:, -1, :])
        # Set placeholder for W and b for calculating the prediction
        # p_W = tf.placeholder(dtype=tf.float32, shape=[hidden_units_2, class_num], name='out_predict_Weights')
        # p_b = tf.placeholder(dtype=tf.float32, shape=[class_num], name='out_predict_bias')
        # Calculate the y predict
        # y_pre = tf.nn.softmax(tf.matmul())
        # Add a dense layer to calculate the predicted logits
        logits = tf.layers.dense(inputs=h_state, units=class_num, name='dense_1')
        # Add a softmax to calculte the predicted y
        y_pred = tf.nn.softmax(logits, name='softmax_1')
        return y_pred

    def __build_rnn_model(self, X):
        """
        Build RNN model for train, validation and predict
        :return: RNN model
        """
        rnn_cell_1 = tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_units_1, name='rnn_1')
        rnn_cell_1 = tf.nn.rnn_cell.DropoutWrapper(cell=rnn_cell_1, input_keep_prob=1.0, output_keep_prob=dropout_ratio)
        rnn_cell_2 = tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_units_2, name='rnn_2')
        rnn_cell_2 = tf.nn.rnn_cell.DropoutWrapper(cell=rnn_cell_2, input_keep_prob=1.0, output_keep_prob=dropout_ratio)
        multi_layer_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell_1, rnn_cell_2], state_is_tuple=True)
        ini_state = multi_layer_cell.zero_state(batch_size=batch_size, dtype=tf.float32)
        output_hidden_states, last_states = tf.nn.dynamic_rnn(multi_layer_cell, inputs=X, initial_state=ini_state, time_major=False)
        h_state = tf.nn.softmax(output_hidden_states[:, -1, :])
        logits = tf.layers.dense(inputs=h_state, units=class_num, name='dense_1')
        y_pred = tf.nn.softmax(logits, name='softmax_1')
        return y_pred

    def __build_gru_model(self, X):
        """
        Build GRU model for train, validation and predict
        :return: GRU model
        """
        gru_cell_1 = tf.nn.rnn_cell.GRUCell(num_units=hidden_units_1, name='gru_1')
        gru_cell_1 = tf.nn.rnn_cell.DropoutWrapper(cell=gru_cell_1, input_keep_prob=1.0, output_keep_prob=dropout_ratio)
        gru_cell_2 = tf.nn.rnn_cell.GRUCell(num_units=hidden_units_2, name='gru_2')
        gru_cell_2 = tf.nn.rnn_cell.DropoutWrapper(cell=gru_cell_2, input_keep_prob=1.0, output_keep_prob=dropout_ratio)
        multi_layer_cell = tf.nn.rnn_cell.MultiRNNCell([gru_cell_1, gru_cell_2], state_is_tuple=True)
        ini_state = multi_layer_cell.zero_state(batch_size=batch_size, dtype=tf.float32)
        output_hidden_states, last_states = tf.nn.dynamic_rnn(multi_layer_cell, inputs=X, initial_state=ini_state, time_major=False)
        h_state = tf.nn.softmax(output_hidden_states[:, -1, :])
        logits = tf.layers.dense(inputs=h_state, units=class_num, name='dense_1')
        y_pred = tf.nn.softmax(logits, name='softmax_1')
        return y_pred

    def fit(self, X_train, y_train,cell='lstm'):
        """
        Fit feature matrix and label matrxi to the Dual lstm model of tensorflow
        :param X: Featrure matrix
        :param y: Label matrix
        :return: Fitted model
        """
        X = tf.placeholder(dtype=tf.float32, shape=(batch_size, time_step, step_vector_size))
        y = tf.placeholder(dtype=tf.float32, shape=(batch_size, class_num))
        if cell == 'lstm':
            y_pred = self.__build_lstm_model(X)
        elif cell == 'rnn':
            y_pred = self.__build_rnn_model(X)
        elif cell == 'gru':
            y_pred = self.__build_gru_model(X)
        # Define the loss
        loss = -tf.reduce_mean(y*tf.log(y_pred))
        train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
        init_op = tf.initialize_all_variables()
        corrections = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(corrections, dtype=tf.float32), name="accuracy")
        batch_num = int(X_train.shape[0]/batch_size)
        with tf.Session() as sess:
            sess.run(init_op)
            for i in range(epochs):
                for j in range(batch_num):
                    inp = X_train[j*batch_size:(j+1)*batch_size]
                    out = y_train[j*batch_size:(j+1)*batch_size]
                    if (j+1) % 4 == 0:
                        train_acc = sess.run(accuracy, feed_dict={X: inp, y: out})
                        print("Epoch - {}, Batch - {}, Train-Acc {}".format(i, j, train_acc))
                    sess.run(train_op, feed_dict={X: inp, y: out})
                    print("Traing {}th epoch, {}th batch".format(i, j))


if __name__ == "__main__":
    spe_universe = DataIO.StockRawData.get_universe()
    strategy = LSTM4Regression(list(spe_universe.code), start_date='2018-01-03', end_date='2018-05-26')
    X, y = strategy.get_train_data_fn()
    strategy.fit(X, y, cell='gru')

