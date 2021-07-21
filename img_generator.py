from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator, array_to_img
from PIL import Image
from numpy import expand_dims


class ImageProcess:
    def __init__(self, path=None, name=None, num=None, dir_path=None):
        self.image = Image.open(path)
        self.size = (224, 224)
        self.path = path
        self.name = name
        self.num = num
        self.dir_path = dir_path

    def image_processing(self):
        self.image = self.image.resize(self.size)
        image_array = np.asarray(self.image)
        image_array = image_array / 255.0
        return image_array

    def image_generator(self):
        img = Image.open(self.path)
        img = img.resize(self.size)
        data = img_to_array(img)
        samples = expand_dims(data, 0)
        datagen = ImageDataGenerator(width_shift_range=[-30, 30],
                                     horizontal_flip=True,
                                     vertical_flip=True)
        it = datagen.flow(samples, batch_size=1)
        for i in range(9):
            batch = it.next()
            array = batch[0].astype('uint8')
            img = array_to_img(array)
            img.save(self.dir_path + self.name + '_' + str(self.num) + '.jpg')
            self.num = self.num+1
        datagen = ImageDataGenerator(height_shift_range=[-30, 30],
                                     horizontal_flip=True,
                                     vertical_flip=True)
        it = datagen.flow(samples, batch_size=1)
        for i in range(9):
            batch = it.next()
            array = batch[0].astype('uint8')
            img = array_to_img(array)
            img.save(self.dir_path + self.name + '_' + str(self.num) + '.jpg')
            self.num = self.num+1
        datagen = ImageDataGenerator(rotation_range=90)
        it = datagen.flow(samples, batch_size=1)
        for i in range(9):
            batch = it.next()
            array = batch[0].astype('uint8')
            img = array_to_img(array)
            img.save(self.dir_path + self.name + '_' + str(self.num) + '.jpg')
            self.num = self.num+1
        # datagen = ImageDataGenerator(brightness_range=[0.2, 1.0])
        # it = datagen.flow(samples, batch_size=1)
        # for i in range(9):
        #     batch = it.next()
        #     array = batch[0].astype('uint8')
        #     img = array_to_img(array)
        #     img.save(self.dir_path + self.name + '_' + str(self.num) + '.jpg')
        #     self.num = self.num+1
        datagen = ImageDataGenerator(zoom_range=[0.5, 1.0],
                                     horizontal_flip=True,
                                     vertical_flip=True)
        it = datagen.flow(samples, batch_size=1)
        for i in range(9):
            batch = it.next()
            array = batch[0].astype('uint8')
            img = array_to_img(array)
            img.save(self.dir_path + self.name + '_' + str(self.num) + '.jpg')
            self.num = self.num+1
        return self.num