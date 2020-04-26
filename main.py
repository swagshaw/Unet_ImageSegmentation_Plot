
from keras.utils import  plot_model
from model import *
from data import *
from matplotlib import pyplot as plt
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
myGene = trainGenerator(2,'data/membrane/train','image','label',data_gen_args,save_to_dir = None)

model = unet()
model.summary()
plot_model(model, to_file='my_unet.png',)
# model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)
history = model.fit_generator(myGene,steps_per_epoch=10,epochs=30)# ,callbacks=[model_checkpoint]
testGene = testGenerator("data/membrane/test")
results = model.predict_generator(testGene,30,verbose=1)
saveResult("data/membrane/test",results)
history_dict = history.history
acc = history_dict['accuracy']
loss = history_dict['loss']
plt.plot(range(1,len(acc)+1),acc ,'b--')
plt.plot(range(1, len(loss)+1), loss, 'r-')
plt.legend(['accuracy', 'loss'])
plt.show()