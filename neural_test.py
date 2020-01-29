#
# Demonstration of a NN that detects an exponential pulse, and its frequency.
#
import numpy as np
import tensorflow as tf
#
tf.enable_eager_execution()
#
from tensorflow import keras
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Dense, Input, Flatten, Conv1D
from tensorflow.nn import relu, softmax, tanh
from IPython.display import SVG
from tensorflow.keras.utils import plot_model

signal_length = 256
NOISE_MAX = 0.3
FREQUENCY_BIN_COUNT = 12
MAX_TIME = 3.0
min_freq, max_freq = (1,5)
min_decay, max_decay = (-1.0,-0.1)
DECAY_BINS = np.geomspace(min_decay,max_decay, num=FREQUENCY_BIN_COUNT+1)

############################################  Build NN Model ####################################### 

inputs = Input(shape=(signal_length, 1,))
    
freq_lay = Conv1D(filters=12, kernel_size=signal_length//4, strides=3, padding='same')(inputs)
freq_lay = Flatten()(freq_lay)
freq_lay = Dense(50, activation=relu, name='frequency_layer')(freq_lay)

decay_lay = Conv1D(filters=12, kernel_size=signal_length//4, strides=3, padding='same')(inputs)
decay_lay = Flatten()(decay_lay)
decay_lay = Dense(50, activation=tanh, name='decay_layer')(decay_lay)

freq_output = Dense(1, activation=relu, name='frequency_output')(freq_lay)
decay_output = Dense(1, activation=tanh, name='decay_output')(decay_lay)

model = tf.keras.Model(
    inputs=inputs,
    outputs=[freq_output, decay_output],
    name="radio detection network")


##############################################  Compile Model ###################################### 

losses = {"frequency_output": "mean_squared_error", 
          "decay_output": "mean_squared_error"}
lossWeights = {"frequency_output": 0.5, 
               "decay_output": 0.5}
Metrics = {"frequency_output": "mean_squared_error", 
          "decay_output": "mean_squared_error"}

#use stochastic gradient descent for optimization
opt = SGD(lr=0.01, momentum=0.9)

model.compile(loss=losses, loss_weights=lossWeights,
            optimizer=opt, 
            metrics=Metrics)

############################################## Get a dataset #######################################


def f(MAX_TIME, t0, signal_length, omega, decay):
    """
    Given a time range MAX_TIME, generate a signal that runs for MAX_TIME that is 
    made up of "signal_length" points, in which a pulse characterized by omega and decay
    is generated, with random amplitude noise imposed over it.
    """
    noise_amplitude = np.random.uniform(0.0, NOISE_MAX)
    noise = np.random.normal(0, noise_amplitude, signal_length)
    t = np.linspace(0, MAX_TIME, signal_length)
    dat = noise + np.heaviside(t - t0, 1)* np.cos(omega*(t - t0))*np.exp(decay*(t - t0))
    return dat.reshape(signal_length, 1)

def guess(MAX_TIME, t0, signal_length, omega, decay):
    t = np.linspace(0, MAX_TIME, signal_length)
    dat = np.heaviside(t - t0, 1)* np.cos(omega*(t - t0))*np.exp(decay*(t - t0))
    return dat

def data_generator(MAX_TIME, min_freq, max_freq, min_decay, max_decay, signal_length, FREQUENCY_BIN_COUNT):
    """
    A generator object that when called makes a signal from f with a random frequency component 
    between the range of min_freq and max_freq, and lambda that is in the range min_lambda and max_lambda
    """
    while True:
        #generate frequency and delay
        omega = np.random.uniform(min_freq, max_freq)
        mylambda = np.random.uniform(min_decay,max_decay)
        t0 = np.random.uniform(0, MAX_TIME//2)
        fake_data = f(MAX_TIME,t0,signal_length, omega, mylambda)

        labels = {
            "frequency_output": [omega],
            "decay_output": [mylambda], 
        }
        
        yield fake_data, labels


if __name__=="__main__":
    print(model.summary())
    print("test")
    if True:
        ds = tf.data.Dataset.from_generator(
            generator=data_generator, 
            output_types=(tf.float32, {"frequency_output":tf.float32,
                                   "decay_output":tf.float32}),  
            output_shapes=((signal_length,1,), {"frequency_output":(1,),
                                   "decay_output":(1,)}),
            args=(MAX_TIME, min_freq, max_freq, min_decay, max_decay, signal_length, FREQUENCY_BIN_COUNT))
        ds = ds.batch(32)

    ######################## Fit the Model ############################## 
        print("TRAINING FREQUENCY MODEL")
        history = model.fit(ds, epochs=7, steps_per_epoch=200, shuffle=True)
        model.save_weights('test.h5')
    else:
        model.load_weights('test.h5')
        
    ################ Use the model to make predictions ########################### 
    import matplotlib.pyplot as plt

    test_data = []
    n_plots = 5
    for i in range(n_plots):
        omega = np.random.uniform(min_freq, max_freq)
        my_lambda = np.random.uniform(min_decay, max_decay)
        t0 = np.random.uniform(0, MAX_TIME//2)
        dat = f(MAX_TIME, t0, signal_length, omega, my_lambda)
        
        [freq_predictions, decay_predictions] = model.predict(np.array([dat]))
        freq_bin = np.argmax(freq_predictions)
        decay_bin = np.argmax(decay_predictions)
        
        print("Guessed Freq  {}".format(freq_predictions[0][0]))
        print("True Freq {}".format(omega))
        print("Guessed Decay  {}".format(decay_predictions))
        print("True Decay {}".format(my_lambda))

        #opening file to store guessed frequency/decay vs real frequency/decay
        
        
        #plotting signal guess
        plt.plot(dat,label="real burst (includes noise)")
        guessdata = guess(MAX_TIME, t0, signal_length,freq_predictions[0][0],decay_predictions[0][0])
        plt.plot(guessdata,label="neural network guessed burst form")
        #plotting actual signal
        realdata = guess(MAX_TIME,t0, signal_length, omega, my_lambda)
        plt.plot(realdata,label="real burst form (noise removed)",linestyle=':',linewidth=5)
        plt.legend()
        plt.title("Complex exponential with {} decay and {} frequency".format(str(my_lambda)[0:4],str(omega)[0:4]))
        plt.savefig("{}_{}.png".format(str(my_lambda)[1:4],str(omega)[0:4]))
        plt.show()
        plt.close()
        
        #plot tensorflow graph
        plot_model(model, to_file='tensor_graph.png')
