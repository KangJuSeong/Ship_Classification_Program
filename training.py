from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import EfficientNetB0
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.python.client import device_lib
import os
import numpy as np
from PIL import Image
from datetime import datetime
import json
from img_generator import ImageProcess
# gpu name : /device:GPU:0


class EfficientNet:
    def __init__(self, epoch, batch_size, classes):
        self.epochs = epoch
        self.batch_size = batch_size
        self.classes = classes
        os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

    def training_model(self, train_input, train_target, test_input, test_target):
        device_lib.list_local_devices()
        tf.config.list_physical_devices('GPU')
        model = EfficientNetB0(include_top=True,
                               weights=None,
                               input_shape=(224, 224, 3),
                               classes=self.classes,
                               classifier_activation='softmax')

        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer=Adam(lr=0.00005),
                      metrics=['acc'])

        history = model.fit(train_input,
                            train_target,
                            validation_data=(test_input, test_target),
                            epochs=self.epochs,
                            batch_size=self.batch_size)

        save_name = str(datetime.today().year) + str(datetime.today().month) + str(datetime.today().day) + str(datetime.today().hour) + str(datetime.today().minute)
        model.save(os.getcwd().replace('\\', '/') + '/save_model/ship_classfication_v' + save_name + '.h5')

        history_dict = history.history
        json.dump(history_dict, open(os.getcwd().replace('\\', '/') + '/acc_history/acc_history_v' + save_name + '.txt', 'w'))

        class_list = os.listdir(os.getcwd().replace('\\', '/') + '/dataset/gen_img')
        _file = open(os.getcwd().replace('\\', '/') + '/class_history/class_history_v' + save_name + '.txt', 'w')
        for i in class_list:
            _file.write(i + '\n')
        _file.close()

    def data_setting(self):
        base_path = os.getcwd().replace('\\', '/') + '/dataset/gen_img'
        data_target = []
        total_img_len = 0
        ready_ship = os.listdir(base_path)
        for i in range(self.classes):
            ship_img_len = len(os.listdir(base_path + '/' + ready_ship[i] + '/'))
            for k in range(ship_img_len):
                data_target.append(i)
            total_img_len = total_img_len + ship_img_len
        data_target = np.array(data_target)
        data_input = np.ndarray(shape=(total_img_len, 224, 224, 3), dtype=np.float32)
        index = 0
        image_size = (224, 224)
        for i in ready_ship:
            path = base_path + '/' + i + '/'
            img_name = os.listdir(path)
            for name in img_name:
                image = Image.open(path + name)
                image = image.resize(image_size)
                image = np.asarray(image)
                image = image / 255.0
                data_input[index] = image
                index = index + 1
                print('이미지 넣는중 {0}/{1}'.format(index, total_img_len))
        train_input, test_input, train_target, test_target = train_test_split(data_input, data_target, test_size=0.2,
                                                                              shuffle=True, stratify=data_target,
                                                                              random_state=32)
        return train_input, test_input, train_target, test_target

    @staticmethod
    def draw_graph(filename):
        with open(os.getcwd().replace('\\', '/') + '/acc_history/'+filename, 'r') as json_file:
            data = json.load(json_file)
            acc = data['acc']
            val_acc = data['val_acc']
            loss = data['loss']
            val_loss = data['val_loss']

            epochs = range(len(acc))
            plt.figure(figsize=(100, 100))
            plt.subplot(211)
            plt.plot(epochs, acc, 'r', label='Training accuracy')
            plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
            plt.title('Training and validation accuracy')
            plt.legend(loc=0)

            plt.subplot(212)
            plt.plot(epochs, loss, 'r', label='Training loss')
            plt.plot(epochs, val_loss, 'b', label='Validation loss')
            plt.title('Training and validation loss')
            plt.legend(loc=0)

            plt.show()

    @staticmethod
    def predict_ship(image_array, model_name, ship_list):
        ship_data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        ship_data[0] = image_array
        model = tf.keras.models.load_model(model_name, compile=False)
        prediction = model.predict(ship_data)
        idx = np.argmax(prediction[0])
        result = '{0} -> {1:0.2f}%'.format(ship_list[idx], prediction[0][idx])
        return result

