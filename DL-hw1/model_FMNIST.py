import tensorflow as tf
from tensorflow.keras import layers

class ResBlock(tf.keras.layers.Layer):

    def __init__(self, filter_num, stride=1):
        super().__init__()
        self.stride = stride

        # Both self.conv1 and self.down_conv layers downsample the input when stride != 1
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv1 = tf.keras.layers.Conv2D(filters=filter_num,
                                            kernel_size=(3, 3),
                                            strides=stride,
                                            padding="same")
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters=filter_num,
                                            kernel_size=(3, 3),
                                            padding="same")

        if self.stride != 1:
            self.down_conv = tf.keras.layers.Conv2D(filters=filter_num,
                                                    kernel_size=(1, 1),
                                                    strides=stride,
                                                    padding="same")
            self.down_bn = tf.keras.layers.BatchNormalization()

    def __call__(self, x, is_training=True):
        identity = x
        if self.stride != 1:
            identity = self.down_conv(identity)
            identity = self.down_bn(identity, training=is_training)

        x = self.bn1(x, training=is_training)
        x = tf.nn.relu(x)
        x = self.conv1(x)
        
        
        x = self.bn2(x, training=is_training)
        x = tf.nn.relu(x)
        x = self.conv2(x)

        return x + identity

def get_model(input_shape=(28,28,1),num_classes=10,units=[32,64],kernel_size=(3,3), kernel_regularizer=None,
		padding='same',activation="relu",kernel_initializer="he_normal", use_ResBlock=True):
	prev = 0
	model = tf.keras.models.Sequential()
	model.add(tf.keras.Input(shape=input_shape))
	model.add(layers.Conv2D(units[0], kernel_size=(3, 3), padding = padding, use_bias=False, kernel_initializer=kernel_initializer,activation=activation, kernel_regularizer=kernel_regularizer))
	model.add(layers.BatchNormalization())
	model.add(layers.MaxPooling2D(pool_size=(2, 2)))
	model.add(layers.Conv2D(units[1], kernel_size=(3, 3), padding = padding, use_bias=False, kernel_initializer=kernel_initializer,activation=activation, kernel_regularizer=kernel_regularizer))
	model.add(layers.BatchNormalization())
	model.add(layers.Activation(activation))
	model.add(layers.MaxPool2D(pool_size=3, strides=2, padding="same"))
	if use_ResBlock:
		for filters in [32] * 4 + [64] * 5 + [128] * 6 + [256] * 3:
			if prev == filters:
				strides = 1 
			else:
				strides = 2 
			model.add(ResBlock(filters, stride=strides))
			prev = filters
	model.add(layers.GlobalAvgPool2D())
	model.add(layers.Flatten())
	model.add(layers.Dense(num_classes, activation="softmax"))
	return model
	
