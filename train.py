


import numpy as np 
from dataset import Dataset
from model import create_3D_model
from keras.optimizers import SGD, Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger, LearningRateScheduler, ReduceLROnPlateau
import time 
import os 

def train_model(epochs, validation_steps, steps_per_epoch):

    c3d = create_3D_model()
    data = Dataset(frames_per_step=16, image_shape=(112,112), batch_size=2, sub_folder='UCF_2')
    train_generator, validation_generator = data.get_generators()
    

    # Get local time.
    time_str = time.strftime("%y%m%d%H%M", time.localtime())
    name_str = None
    if name_str == None:
        name_str = time_str

    # Callbacks: Save the model.
    directory1 = os.path.join('out', 'checkpoints', name_str)
    if not os.path.exists(directory1):
        os.makedirs(directory1)
        checkpointer = ModelCheckpoint(
            filepath=os.path.join(directory1, '{epoch:03d}-{val_loss:.3f}.hdf5'),
            verbose=1,
            save_best_only=True)

    # Callbacks: TensorBoard
    directory2 = os.path.join('out', 'TB', name_str)
    if not os.path.exists(directory2):
        os.makedirs(directory2)
    tb = TensorBoard(log_dir=os.path.join(directory2))

    # Callbacks: Early stoper
    early_stopper = EarlyStopping(monitor='loss', patience=100)

    # Callbacks: Save results.
    directory3 = os.path.join('out', 'logs', name_str)
    if not os.path.exists(directory3):
        os.makedirs(directory3)
    timestamp = time.time()
    csv_logger = CSVLogger(os.path.join(directory3, 'training-' + \
        str(timestamp) + '.log'))

    lr_reduce = ReduceLROnPlateau(monitor='val_acc',
                                    patience=40,
                                    verbose=1,
                                    factor=0.5,
                                    min_lr=0)

    c3d.compile(
        optimizer=SGD(lr=0.003, momentum=0.9),
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    c3d.fit_generator(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        epochs=epochs,
        callbacks=[tb, early_stopper, csv_logger, checkpointer, lr_reduce])

    


if __name__ == '__main__':

    epochs = 100
    validation_steps = 5
    steps_per_epoch = 10


    train_model(epochs=epochs, validation_steps=validation_steps, steps_per_epoch=steps_per_epoch)

