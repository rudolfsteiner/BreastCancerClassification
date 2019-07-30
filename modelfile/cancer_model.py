import tensorflow as tf

class CancerModel:

    @staticmethod
    
    def build(width, height, depth, classes):
        
        def _conv_block(inputs, conv_type, filters, kernel_size, strides, conv_name, padding='same', 
                relu=True):
  
            if(conv_type == 'ds'):
                x = tf.keras.layers.SeparableConv2D(filters, kernel_size, padding=padding, 
                                                    strides = strides, name = "ds_" + conv_name)(inputs)
            else:
                x = tf.keras.layers.Conv2D(filters, kernel_size, padding=padding, 
                                           strides = strides, name = "conv_" + conv_name)(inputs)  

            x = tf.keras.layers.BatchNormalization()(x)

            if (relu):
                x = tf.keras.activations.relu(x)

            return x        
    
            
        input_layer = tf.keras.layers.Input(shape=(height, width, depth), name = 'input_layer')

        # Conv layers

        conv1_layer = _conv_block(input_layer, conv_type = 'ds', filters= 32, 
                                  kernel_size = (3, 3), strides = (1,1), conv_name = "conv1")
        conv1_layer = tf.keras.layers.MaxPooling2D(pool_size = (2,2))(conv1_layer)
        conv1_layer = tf.keras.layers.Dropout(0.25)(conv1_layer)
     
        conv2_layer = _conv_block(conv1_layer, conv_type = 'ds', filters=64, 
                                  kernel_size = (3, 3), strides = (1,1), conv_name = "conv2")
        
        conv3_layer = _conv_block(conv2_layer,conv_type = 'ds', filters=64, 
                                  kernel_size = (3, 3), strides = (1,1), conv_name = "conv3")
        conv3_layer = tf.keras.layers.MaxPooling2D(pool_size = (2,2))(conv3_layer)
        conv3_layer = tf.keras.layers.Dropout(0.25)(conv3_layer)

        conv4_layer = _conv_block(conv3_layer, conv_type = 'ds', filters=64, 
                                  kernel_size = (3, 3), strides = (1,1), conv_name = "conv4")
        
        conv5_layer = _conv_block(conv4_layer, conv_type = 'ds', filters=64, 
                                  kernel_size = (3, 3), strides = (1,1), conv_name = "conv5")

        conv6_layer = _conv_block(conv5_layer, conv_type = 'ds', filters=64, 
                                  kernel_size = (3, 3), strides = (1,1), conv_name = "conv6")
        conv6_layer = tf.keras.layers.MaxPooling2D(pool_size = (2,2))(conv6_layer)
        conv6_layer = tf.keras.layers.Dropout(0.25)(conv6_layer)

        # Classifiers
        
        classifier = tf.keras.layers.Flatten()(conv6_layer)
        classifier = tf.keras.layers.Dense(256)(classifier)
        classifier = tf.keras.layers.Activation("relu")(classifier)
        classifier = tf.keras.layers.BatchNormalization()(classifier)
        classifier = tf.keras.layers.Dropout(0.5)(classifier)
        classifier = tf.keras.layers.Dense(classes)(classifier)
        classifier = tf.keras.layers.Activation("softmax")(classifier)
        
        model = tf.keras.Model(inputs = input_layer , outputs = classifier, name = 'CancerModel')

        return model