import tflearn
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

# 상수 정의
n_inputs = 4
n_hidden1 = 1000
n_hidden2 = 1000
n_outputs = 3
# 학습 데이터 읽어 오기
iris = datasets.load_iris()
x_data = iris.data
y_data = iris.target
# 입력 데이터 스케일링
minmax_scaler = MinMaxScaler()
minmax_scaler.fit(x_data)
x_data = minmax_scaler.transform(x_data)

# 데이터 학습용과 테스트용으로 분리
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=3, stratify=y_data)
enc = OneHotEncoder(handle_unknown='ignore')
y_train = enc.fit_transform(y_train.reshape(-1,1)).toarray()
y_test = enc.fit_transform(y_test.reshape(-1,1)).toarray()

# 학습 패러미터 설정
n_epochs = 100; batch_size = 5

# 망 구성
inputs = tflearn.input_data(shape=[None, n_inputs])
hidden1 = tflearn.fully_connected(inputs, n_hidden1, activation='elu', name='hidden1')
hidden2 = tflearn.fully_connected(hidden1 , n_hidden2, activation='elu', name='hidden2')
softmax = tflearn.fully_connected(hidden2, n_outputs, activation='softmax', name ='output')
net = tflearn.regression(softmax)

# 모델 객체 생성
model = tflearn.DNN(net)

# 모델 학습
model.fit(x_train, y_train, validation_set=None, n_epoch=n_epochs,batch_size=batch_size)

# 성능 평가
acc_train = model.evaluate(x_train, y_train, batch_size)
acc_test = model.evaluate(x_test, y_test, batch_size)
print("학습 데이터 : " + str(acc_train) + ", " + "테스트 데이터 : " + str(acc_test))
