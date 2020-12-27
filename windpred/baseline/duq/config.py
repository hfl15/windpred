class ParameterConfig(object):
    def __init__(self):
        self.num_input_features = 1  # Constant variable
        self.num_output_features = 1  # Constant variable
        self.num_decoder_features = 1  # Constant variable
        self.input_sequence_length = 24  # Constant variable
        self.target_sequence_length = 24  # Constant variable (order strict)

        self.num_steps_to_predict = 24
        self.num_stations = 9
        self.num_variables_to_predict = 1
        self.num_statistics_to_predict = 2  # predict both mean and variance

        self.layers = [50, 50]
        layers_str = ''
        for i in self.layers:
            layers_str += str(i) + '_'

        self.lr = 0.001
        self.decay = 0
        self.loss = 'mae'  # must be mve, mse, OR mae
        self.early_stop_limit = 10  # with the unit of Iteration Display
        self.pi_dic = {0.95: 1.96, 0.9: 1.645, 0.8: 1.28,
                       0.68: 1.}  # Gaussian distribution confidence interval ({confidence:variance})
        self.regulariser = None
        self.dropout_rate = 0

        self.id_embd = True
        self.time_embd = True
        self.model_name_format_str = '_layers_{}loss_{}_dropout{}'.format(layers_str, self.loss, self.dropout_rate)
