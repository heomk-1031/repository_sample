import tensorflow as tf


class SimpleDense(tf.keras.Model):
    def __init__(self, units, num_layers, hidden_activation, output_dim, output_activation, **kwargs):
        super(SimpleDense, self).__init__(**kwargs)
        self.units = units
        self.num_layers = num_layers
        self.hidden_activation = hidden_activation
        self.output_dim = output_dim
        self.output_activation = output_activation
        self.flatter = tf.keras.layers.Flatten()
        self.hidden_layers = [tf.keras.layers.Dense(units,
                                                    activation=tf.keras.activations.get(hidden_activation)) 
                              for _ in range(num_layers)]
        self.output_layer = tf.keras.layers.Dense(output_dim, tf.keras.activations.get(output_activation))
                
    def call(self, inputs):
        if len(tf.shape(inputs)) != 1:
            x = self.flatter(inputs)
        for n in range(self.num_layers):
            x = self.hidden_layers[n](x)
        outputs = self.output_layer(x)
        return outputs
        

class SimpleConv(tf.keras.Model):
    def __init__(self, filters, num_layers, kernel_size, hidden_activation, output_dim, output_activation, dropout_rate, **kwargs):
        super(SimpleConv, self).__init__(**kwargs)
        self.filters = filters
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.hidden_activation = hidden_activation
        self.output_dim = output_dim
        self.output_activation = output_activation
        self.dropout_rate = dropout_rate
        self.hidden_layers = [tf.keras.layers.Conv2D(filters,
                                                     kernel_size,
                                                     strides=(1,1),
                                                     padding='same',
                                                     activation = tf.keras.activations.get(hidden_activation)) 
                              for _ in range(num_layers)]
        self.hidden_bns = [tf.keras.layers.BatchNormalization() for _ in range(num_layers)]
        self.drop_outs = [tf.keras.layers.Dropout(dropout_rate) for _ in range(num_layers)]        
        self.max_pools = [tf.keras.layers.MaxPool2D() for _ in range(num_layers)]
        self.global_avg_pool = tf.keras.layers.GlobalAvgPool2D()
        self.output_layer = tf.keras.layers.Dense(output_dim, tf.keras.activations.get(output_activation))
        
        
    def call(self, inputs):
        x = inputs
        for n in range(self.num_layers):
            x = self.hidden_layers[n](x)
            x = self.drop_outs[n](x)
            x = self.hidden_bns[n](x)
            x = self.max_pools[n](x)
        
        x = self.global_avg_pool(x)
        outputs = self.output_layer(x)
        return outputs
        
        


if __name__ == "__main__":
    print('Moduel Testing')
    test_input = tf.random.uniform(shape=(2, 32, 32, 1))
    
    simple_dense_model = SimpleDense(10, 2, 'relu', 10, 'softmax')
    simple_dense_output = simple_dense_model(test_input)
    print(simple_dense_output)
    
    simple_conv_model = SimpleConv(10, 2, 3, 'relu', 10, 'softmax', 0.2)
    simple_conv_output = simple_conv_model(test_input)
    print(simple_conv_output)
    
    