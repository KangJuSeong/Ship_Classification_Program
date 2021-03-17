import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import requests
from img_generator import ImageProcess as imgP
import os
import urllib.request as rq
from training import EfficientNet as eF
import time
import matplotlib.pyplot as plt
from datetime import datetime
import base64
import numpy as np
from PIL import Image
import random


class TrainingProgram(QWidget):

    def __init__(self):
        super().__init__()
        self.base_url = 'http://127.0.0.1:8000/'
        self.ship_label = QLabel('학습 선박 : ', self)
        self.ship_edit = QLineEdit(self)
        self.info_label = QLabel('선박 정보 : ', self)
        self.gen_state_label = QLabel('이미지 증식 진행률 : ', self)
        self.cb_waste = QCheckBox('유기,폐 선박', self)
        self.cb_normal = QCheckBox('일반 선박', self)
        self.load_ship = QPushButton('선박 불러오기', self)
        self.image_gen = QPushButton('이미지 증식하기', self)
        self.gen_state = QLabel('상태 : ', self)
        self.gen_bar = QProgressBar(self)
        self.info = QLabel('', self)
        self.train_list_label = QLabel('학습 된 선박', self)
        self.un_train_list_label = QLabel('학습 준비 선박', self)
        self.train_list = QListView(self)
        self.un_train_list = QListView(self)
        self.refresh = QPushButton('새로고침', self)
        self.train = QPushButton('학습하기', self)
        self.train_state = QLabel('학습 상태 : 준비중', self)
        self.model = QStandardItemModel()
        self._model = QStandardItemModel()
        self.graph_btn = QPushButton('결과 그래프', self)
        self.graph_label = QLabel('그래프 그리기 : ', self)
        self.graph_load = QPushButton('파일 불러오기', self)
        self.graph_name = QLabel('', self)
        self.epoch = QLineEdit(self)
        self.epoch_label = QLabel('에포크 : ', self)
        self.batch_size = QLineEdit(self)
        self.batch_size_label = QLabel('배치 : ', self)
        self.predict_btn = QPushButton('예측하기', self)
        self.load_model_btn = QPushButton('모델 불러오기', self)
        self.load_model_label = QLabel('모델명 : ', self)
        self.ship_idx_label = QLabel('이미지 번호 : ', self)
        self.ship_idx = QLineEdit(self)
        self.result = QLabel('결과 : ', self)
        self.train_ship_list = []
        self.un_train_ship_list = os.listdir(os.getcwd().replace('\\', '/') + '/dataset/gen_img')
        self.img_list = []
        self.img_name = ''
        self.test_path = ''
        self.train_path = ''
        self.gen_path = ''
        self.model_path = ''
        file_list = os.listdir(os.getcwd().replace('\\', '/') + '/class_history')
        file1 = open(os.getcwd().replace('\\', '/') + '/class_history/' + file_list[-1], 'r')
        lines = file1.readlines()
        for line in lines:
            self.train_ship_list.append(line.rstrip('\n'))
        if self.train_ship_list[0] == '':
            self.train_ship_list = list()
        self.un_train_ship_list = list(set(self.un_train_ship_list) - set(self.train_ship_list))
        self.initUI()

    def initUI(self):
        self.ship_label.move(50, 50)
        self.ship_edit.resize(400, 30)
        self.ship_edit.move(120, 50)
        self.info_label.move(50, 100)
        self.info.move(150, 100)
        self.info.resize(300, 30)
        self.gen_state_label.move(50, 150)
        self.gen_state_label.resize(150, 30)
        self.cb_waste.move(550, 50)
        self.cb_normal.move(650, 50)
        self.load_ship.move(750, 50)
        self.load_ship.clicked.connect(self.load_ship_data)
        self.image_gen.move(750, 100)
        self.image_gen.clicked.connect(self.generator_img)
        self.gen_state.move(800, 150)
        self.gen_bar.move(170, 150)
        self.gen_bar.resize(600, 30)
        self.train_list_label.move(50, 220)
        self.un_train_list_label.move(300, 220)
        self.train_list.move(50, 250)
        self.train_list.resize(200, 400)
        self.un_train_list.move(300, 250)
        self.un_train_list.resize(200, 400)
        self.train.move(825, 295)
        self.epoch.move(650, 295)
        self.epoch_label.move(600, 300)
        self.epoch.resize(50, 20)
        self.batch_size.move(750, 295)
        self.batch_size.resize(50, 20)
        self.batch_size_label.move(710, 300)
        self.refresh.move(600, 250)
        self.refresh.clicked.connect(self.refresh_screen)
        self.train.clicked.connect(self.train_ship)
        for ship in self.train_ship_list:
            self.model.appendRow(QStandardItem(ship))
        self.train_list.setModel(self.model)
        for ship in self.un_train_ship_list:
            self._model.appendRow(QStandardItem(ship))
        self.un_train_list.setModel(self._model)
        self.train_state.move(600, 330)
        self.train_state.resize(200, 50)
        self.graph_btn.move(820, 410)
        self.graph_btn.clicked.connect(self.draw_graph)
        self.graph_load.move(700, 410)
        self.graph_load.clicked.connect(self.load_graph)
        self.graph_label.move(600, 385)
        self.graph_name.move(700, 385)
        self.graph_name.resize(200, 15)
        self.load_model_label.move(600, 470)
        self.load_model_label.resize(300, 15)
        self.load_model_btn.move(700, 545)
        self.predict_btn.move(820, 545)
        self.predict_btn.clicked.connect(self.predict_ship)
        self.ship_idx_label.move(600, 505)
        self.ship_idx.move(680, 502)
        self.ship_idx.resize(40, 20)
        self.load_model_btn.clicked.connect(self.load_model)
        self.result.move(600, 600)
        self.result.resize(300., 15)

        self.setGeometry(500, 100, 1000, 800)
        self.setWindowTitle('Auto Training Program')
        self.show()

    def load_ship_data(self):
        if self.cb_normal.isChecked():
            keyword = self.ship_edit.text()
            data = requests.get(self.base_url + 'Ships/ship/normal/program/' + keyword + '/').json()['data']
            self.img_name = data['name']
            img_cnt = data['img_cnt']
            self.info.setText('{2}, {0}개 이미지 보유, 증식 후 {1}개 이미지 생성'.format(img_cnt, (int(img_cnt)-1) * 54, self.img_name))
            self.img_list = data['normal_imgs']
            self.gen_path = os.getcwd().replace('\\', '/') + '/dataset/gen_img/' + self.img_name + '/'
            self.train_path = os.getcwd().replace('\\', '/') + '/dataset/train_img/' + self.img_name + '/'
            self.test_path = os.getcwd().replace('\\', '/') + '/dataset/test_img/' + self.img_name + '/'
        elif self.cb_waste.isChecked():
            keyword = self.ship_edit.text()
            data = requests.get(self.base_url + 'Ships/ship/waste/program/' + keyword + '/').json()['data']
            self.img_name = data['id']
            img_cnt = data['img_cnt']
            self.info.setText('{2}, {0}개 이미지 보유, 증식 후 {1}개 이미지 생성'.format(img_cnt, (int(img_cnt)-1) * 54, self.img_name))
            self.img_list = data['waste_imgs']
            self.gen_path = os.getcwd().replace('\\', '/') + '/dataset/gen_img/' + self.img_name + '/'
            self.train_path = os.getcwd().replace('\\', '/') + '/dataset/train_img/' + self.img_name + '/'
            self.test_path = os.getcwd().replace('\\', '/') + '/dataset/test_img/' + self.img_name + '/'
        else:
            msg = QMessageBox()
            msg.setText("선박 종류를 선택해주세요")
            msg.setWindowTitle("경고")
            retval = msg.exec_()

    def generator_img(self):
        self.gen_bar.setRange(0, len(self.img_list) * 54 + len(self.img_list))
        if not os.path.exists(self.train_path):
            os.makedirs(self.train_path)
        if not os.path.exists(self.test_path):
            os.makedirs(self.test_path)
        if not os.path.exists(self.gen_path):
            os.makedirs(self.gen_path)
        idx = 0
        test_idx = 0
        base_url = self.base_url[:-1]
        self.gen_state.setText('상태 : 진행중')
        test_img_list = []
        test_cnt = int(len(self.img_list) * 0.2)
        for i in range(test_cnt):
            test_img_list.append(self.img_list[random.randint(0, len(self.img_list))])
        for img in self.img_list:
            if img in test_img_list:
                img_data_url = base_url + img
                img_data = requests.get(img_data_url)
                img_path = self.test_path + self.img_name + '_' + str(test_idx) + '.jpg'
                image = open(img_path, 'wb')
                image.write(img_data.content)
                image.close()
                test_idx = test_idx + 1
                continue
            self.gen_bar.setValue(idx)
            img_data_url = base_url + img
            img_data = requests.get(img_data_url)
            img_path = self.gen_path + self.img_name + '_' + str(idx) + '.jpg'
            image = open(img_path, 'wb')
            image.write(img_data.content)
            image.close()
            idx = idx + 1
            obj = imgP(img_path, self.img_name, idx, self.gen_path)
            idx = obj.image_generator()
        self.gen_bar.setValue(self.gen_bar.maximum())
        if self.gen_bar.value() == self.gen_bar.maximum():
            self.gen_state.setText('상태 : 완료')

    def train_ship(self):
        self.train_state.setText('학습 상태 : 학습 중')
        model = eF(epoch=int(self.epoch.text()),
                   batch_size=int(self.batch_size.text()),
                   classes=len(self.train_ship_list)+len(self.un_train_ship_list))
        train_input, test_input, train_target, test_target = model.data_setting()
        model.training_model(train_input=train_input,
                             test_input=test_input,
                             train_target=train_target,
                             test_target=test_target)
        self.train_state.setText('학습 상태 : 학습 완료')

    def refresh_screen(self):
        self.un_train_ship_list = os.listdir(os.getcwd().replace('\\', '/') + '/dataset/img_gen')
        self.train_ship_list = []
        file_list = os.listdir(os.getcwd().replace('\\', '/') + '/class_history')
        file1 = open(os.getcwd().replace('\\', '/') + '/class_history/' + file_list[-1], 'r')
        lines = file1.readlines()
        self._model = QStandardItemModel()
        self.model = QStandardItemModel()
        for line in lines:
            self.train_ship_list.append(line.rstrip('\n'))
        self.un_train_ship_list = list(set(self.un_train_ship_list) - set(self.train_ship_list))
        for ship in self.train_ship_list:
            self.model.appendRow(QStandardItem(ship))
        self.train_list.setModel(self.model)
        for ship in self.un_train_ship_list:
            self._model.appendRow(QStandardItem(ship))
        self.un_train_list.setModel(self._model)

    def draw_graph(self):
        eF.draw_graph(self.graph_name.text())

    def load_graph(self):
        path = QFileDialog.getOpenFileName(self)
        name_idx = path[0].find('history')
        name = path[0][name_idx+8:]
        self.graph_name.setText(name)

    def load_model(self):
        path = QFileDialog.getOpenFileName(self)
        self.model_path = path[0]
        name_idx = path[0].find('save_model')
        name = path[0][name_idx+11:]
        self.load_model_label.setText('모델명 : ' + name)

    def predict_ship(self):
        data = requests.get('http://127.0.0.1:8000/Ships/ship/normal/program/' + self.ship_idx.text()).json()['data']
        img_url = data['main_img']
        img_path = 'http://127.0.0.1:8000' + img_url
        img = Image.open(requests.get(img_path, stream=True).raw)
        img = img.resize((224, 224))
        image_array = np.asarray(img)
        plt.imshow(image_array)
        plt.show()
        image_array = image_array / 255.0
        result = eF.predict_ship(image_array, self.model_path, self.train_ship_list)
        self.result.setText('결과 : ' + result)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = TrainingProgram()
    sys.exit(app.exec_())
