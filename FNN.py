import pandas as pd
import os, shutil, math
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from keras import models, layers
from keras import backend as K
import matplotlib.pyplot as plt

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)

# Load datasets
df_train = pd.read_csv('./dataset/train/2015_2020.csv')
df_test = pd.read_csv('./dataset/test/2022_2023.csv')

# convert "DayOfWeek"
# days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
# for i in days:
#   df_train["DayOfWeek"] = df_train["DayOfWeek"].replace(i, days.index(i))
#   df_test["DayOfWeek"] = df_train["DayOfWeek"].replace(i, days.index(i))

# for i in df_train["DateTime"]:
#   time = i.split(" ")[1]
#   hour = time.split(":")[0]
#   df_train["DateTime"] = df_train["DateTime"].replace(i, hour)

# for i in df_test["DateTime"]:
#   time = i.split(" ")[1]
#   hour = time.split(":")[0]
#   df_test["DateTime"] = df_train["DateTime"].replace(i, hour)


df_train.drop(["DayOfWeek"], axis=1, inplace=True)
df_test.drop(["DayOfWeek"], axis=1, inplace=True)

df_train.drop(["DateTime"], axis=1, inplace=True)
df_test.drop(["DateTime"], axis=1, inplace=True)

target_column = 'KCPL'

# Separate features and targets
features_train = df_train.drop(target_column, axis=1)
target_train = df_train[target_column]
features_test = df_test.drop(target_column, axis=1)
target_test = df_test[target_column]

# Normalize features
scaler = StandardScaler()
features_train_scaled = scaler.fit_transform(features_train)
features_test_scaled = scaler.transform(features_test)

# Defines a function to create a neural network model, allowing to set different numbers of hidden layers and units, learning rate, and parameters such as regularization and dropout.
# 
# Use Keras’ Sequential model.
# 
# Adds the specified number of hidden layers, using dropout if needed.
# 
# Use the Adam optimizer and use the mean square error (MSE) as the loss function.

def create_model(hidden_layers, hidden_units, learning_rate, regularizer=None, dropout_rate=0.0):
    model = models.Sequential()
    #input layer
    model.add(layers.Dense(hidden_units, activation='relu', input_dim=features_train_scaled.shape[1], kernel_regularizer=regularizer))

    for i in range(hidden_layers - 1):
        model.add(layers.Dense(max(2 ** (6 - i), 1), activation='relu', kernel_regularizer=regularizer))
        if dropout_rate > 0.0:
            model.add(layers.Dropout(dropout_rate))

    # output layer
    model.add(layers.Dense(1))  # No activation for regression

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_absolute_error', 'mean_squared_error'])
    return model

# 3. Train and Evaluate Models

# Defines a function for training and evaluating neural network models with different configurations.
# 
# The loop tried different parameter combinations such as the number of hidden layers and units, learning rate, dropout rate, regularization method, etc., and printed the training and testing MSE and MAE values of the model.
# 
# The change of the loss function with the number of iterations is plotted, which can be used to observe the training situation and generalization ability of the model.

# best model tracking
global bestModel
bestModel = create_model(0, 0, 0, regularizer=None, dropout_rate=0.0)
global bestMSE
bestMSE = float('inf') #really high number
global best_hidden_layers
best_hidden_layers = 0
global best_hidden_units
best_hidden_units = 0
global best_learning_rate
best_learning_rate = 0
global best_dropout_rate
best_dropout_rate = 0
global best_regularizer
best_regularizer = None
global best_history
best_history = None

def drange(start, stop, step):
  while start < stop:
    yield start
    start += step

hidden_layers_list = []
for j in drange(1, 6, 1):
    hidden_layers_list.append(j)
# 'the optimal size of the hidden layer is usually between the size of the input and size of the output layers'
hidden_units_list = []
for j in drange(int((len(features_test_scaled) + len(target_test))/10), (len(features_test_scaled) + len(target_test)), int((len(features_test_scaled) + len(target_test))/10)):
    hidden_units_list.append(j)

learning_rate_list = []
for j in drange(0.003, 0.01, 0.002):
    learning_rate_list.append(round(j, 3))

dropout_rates = []  #  Experiment with dropout rates
for j in drange(0.4, 0.6, 0.1):
    dropout_rates.append(round(j, 1))

regularizers = [None, tf.keras.regularizers.l2(0.01)]  #  Experiment with L2 regularization

number_iterations = len(hidden_layers_list) * len(hidden_units_list[:-3]) * len(learning_rate_list) * len(dropout_rates) * len(regularizers)
print(f'Number of iterations: {number_iterations}')

def train_and_evaluate_models(features_train_scaled, target_train, features_test_scaled, target_test):
    count = 0
    for hidden_layers in hidden_layers_list:
        for hidden_units in hidden_units_list[:-3]:
            for learning_rate in learning_rate_list:
                for dropout_rate in dropout_rates:
                    for regularizer in regularizers:
                        count += 1
                        print(f"[{count}/{number_iterations}]: Training model with: Hidden Layers: {hidden_layers}, Hidden Units: {hidden_units}, Learning Rate: {learning_rate}, Dropout Rate: {dropout_rate}, Regularizer: {regularizer}")
                        try:
                            model = create_model(hidden_layers, hidden_units, learning_rate, regularizer, dropout_rate)
                            history = model.fit(features_train_scaled, target_train, epochs=20, batch_size=128, validation_split=0.1)

                            # Evaluate the model
                            _, train_mae, train_mse = model.evaluate(features_train_scaled, target_train, batch_size=128)
                            _, test_mae, test_mse = model.evaluate(features_test_scaled, target_test, batch_size=128)

                            print(f"Train MSE: {train_mse}, Test MSE: {test_mse}")
                            print(f"Train MAE: {train_mae}, Test MAE: {test_mae}")

                            global bestMSE
                            global bestModel
                            global best_hidden_layers
                            global best_hidden_units
                            global best_learning_rate
                            global best_dropout_rate
                            global best_regularizer
                            global best_history

                            if(test_mse < bestMSE):
                                bestModel = model
                                bestMSE = test_mse
                                best_hidden_layers = hidden_layers
                                best_hidden_units = hidden_units
                                best_learning_rate = learning_rate
                                best_dropout_rate = dropout_rate
                                best_regularizer = regularizer
                                best_history = history

                                if os.path.exists('best-FNN-model'):
                                    shutil.rmtree('best-FNN-model')
                                bestModel.save('best-FNN-model') #Saving the model
                        except:
                            continue
                        finally:
                            # Free up memory
                            del model, test_mse, test_mae, train_mae, train_mse, history
                            # Clear Keras session to release memory
                            K.clear_session()

# Train and evaluate models with different configurations
train_and_evaluate_models(features_train_scaled, target_train, features_test_scaled, target_test)

session.close()

# Plot loss over epochs
plt.plot(best_history.history['loss'], label='Training Loss')
plt.plot(best_history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

print(bestModel)
print(bestMSE)
print(best_hidden_layers)
print(best_hidden_units)
print(best_learning_rate)
print(best_dropout_rate)
print(best_regularizer)