
#------------------------------------standard Fonctions for CNN SW Pipelin --------------------------------------------------# 

import tensorflow as tf

# Fonction Normalize

def normalize(x):
    return (x.astype(float) - 128) / 128

#-------------------------------------------
# Fonction evaluate the model
# X_data (input data : # X_data (input data : differents features), y_data (label))
# total_accuracy / num_examples = model accuracy

#make the mean of the accuracy for every batch, and finally caculate the accuracy of all the model

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples
#--------------------------------------------
