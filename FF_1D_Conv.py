#!/usr/bin/env python
# coding: utf-8

# In[35]:


import numpy as np
import keras
from keras.layers import Conv1D, Dense, Dropout, MaxPooling1D, Flatten,GlobalMaxPooling1D
from keras.models import Sequential
from keras import optimizers
from keras.optimizers import Adam 
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt


# In[36]:


class LossHistory(keras.callbacks.Callback):

    # This function is called when the training begins
    def on_train_begin(self, logs={}):
        # Initialize the lists for holding the logs, losses and accuracies
        self.losses = []
        self.acc = []
        self.val_losses = []
        self.val_acc = []
        self.logs = []

    # This function is called at the end of each epoch
    def on_epoch_end(self, epoch, logs={}):

        # Append the logs, losses and accuracies to the lists
        self.logs.append(logs)
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('acc'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_acc'))

        # Before plotting ensure at least 2 epochs have passed
        if len(self.losses) > 1:

            N = np.arange(0, len(self.losses))

            # You can chose the style of your preference
            # print(plt.style.available) to see the available options
            #plt.style.use("seaborn")

            # Plot train loss, train acc, val loss and val acc against epochs passed
            plt.figure()
            plt.plot(N, self.losses, label = "train_loss")
            plt.plot(N, self.val_losses, label = "val_loss")
            plt.title("Training Loss and Validation Loss [Epoch {}]".format(epoch))
            plt.xlabel("Epoch #")
            plt.ylabel("Loss")
            plt.legend()
            # Make sure there exists a folder called output in the current directory
            # or replace 'output' with whatever direcory you want to put in the plots
            plt.show()


# In[37]:


output = np.loadtxt('Absorption_50.txt')
input = np.loadtxt('Thickness_50.txt')

output= np.array(output)
input= np.array(input)
train_input, test_input, train_output, test_output = train_test_split(input, output, test_size=0.2, shuffle= True)
train_input, val_input, train_output, val_output = train_test_split(train_input, train_output, test_size=0.25, shuffle= True)
print(train_input.shape)
print(train_output.shape)


# In[38]:


model = Sequential()

train_input = train_input.reshape(train_input.shape[0], 9, 1).astype('float32')
val_input = val_input.reshape(val_input.shape[0], 9, 1).astype('float32') 
test_input = test_input.reshape(val_input.shape[0], 9, 1).astype('float32')

model = Sequential()
model.add(Conv1D(filters=32, kernel_size=2,use_bias=False, activation='relu',input_shape=(9, 1)))
model.add(MaxPooling1D(pool_size=2 ))
model.add(Conv1D(filters=64, kernel_size=2,use_bias=False,activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(200, activation='relu'))
model.summary()


# In[39]:


adam = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)

model.compile(loss='mse',optimizer='adam' ,metrics=['mse'])


# In[40]:


history = LossHistory()

model.fit(train_input, train_output, batch_size=64, epochs=10, shuffle=True,verbose=2,validation_data=(val_input, val_output),callbacks=[history])


# In[41]:


predicted_output = model.predict(val_input) # get prediction for test_input
errors = list()
for true_val, pred_val in zip(val_output, predicted_output):
    temp_error = metrics.mean_absolute_error(true_val, pred_val) 
    errors.append(temp_error)
errors = np.asarray(errors)


# In[42]:


plt.figure()
x = range(len(errors))
plt.plot(x, errors)
plt.xlabel('Validation Samples')
plt.ylabel('Prediction Error')
plt.show()


# In[43]:


structure=(1,45,4.2,65,10,6.6,4.4,10,6.6) # Proposed Multilayer Structure Layer Thicknesess
structure=np.array(structure).reshape(-1,9,1)
spectrum_predict=model.predict(structure)
spectrum_predict=np.array(spectrum_predict).reshape(200)


# In[44]:


with open('Wavelength_50.txt') as f:
    lines = f.readlines()
    x1 = [line.split()[0] for line in lines]

for i in range(0, len(x1)): 
    x1[i] = float(x1[i]) 

x1 = np.reshape(x1,(200,1)) 
x1 = x1.flatten() 

with open('Wavelength_300_25000_Aem_11.txt') as f:
    lines = f.readlines()
    x2 = [line.split()[0] for line in lines]

for i in range(0, len(x2)): 
    x2[i] = float(x2[i]) 

x2 = np.reshape(x2,(200,1)) 
x2 = x2.flatten() 
with open('Emissivity_300_25000_Aem_11.txt') as f:
    lines = f.readlines()
    y2 = [line.split()[0] for line in lines]

for i in range(0, len(y2)): 
    y2[i] = float(y2[i]) 

y2 = np.reshape(y2,(200,1)) 
y2 = y2.flatten() 

f = plt.figure(figsize=(5,5))
ax = f.add_subplot(211)
ax.plot(x1, abs(spectrum_predict), lw=2, color='blue',label='Predicted')
ax.set_xlabel('Wavelength (um)')
ax.set_ylabel('Emissivity')
ax.grid(True)

ax.plot(x2, y2, lw=2, label='Real', color='green')
ax.set_xlabel('Wavelength (um)')
ax.set_ylabel('Emissivity')
ax.grid(True)
#plt.figure()
#plt.plot(x1, abs(spectrum_predict)) # Absorption Spectrum
#plt.grid(True)
#plt.xlabel('wavelength in um')
#plt.ylabel('spectrum_predict')
#plt.show()
# plt.savefig('Spectrum_predicted_vs_wavelength.jpg')
# plt.close()


# In[45]:


f=open('Predicted_Spectrum.txt','a')
for j in range(200):
    f.write(str(spectrum_predict[j]))
    f.write('  ')

f.close()


# In[ ]:




