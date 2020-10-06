from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
iris_data = iris.data
iris_label = iris.target
iris_train_data, iris_test_data, iris_train_label, iris_test_label = train_test_split(iris_data, iris_label, test_size=0.2, random_state=0, stratify=iris_label)
# stratify = y할 경우 층별 샘플링

scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
scaler.fit(iris_train_data)
scaled_iris_train_data = scaler.transform(iris_train_data)

scaled_iris_test_data = scaler.transform(iris_test_data)
#fit을 한 번 더 적용하면 학습 데이터와 테스트 데이터 간 스케일링이 맞지않아진다.
#ML 모델은 학습 데이터를 기반으로 학습되기때문에 반드시 테스트 데이터는 학습 데이터의 스케일링 기준을 따라야한다.

print("학습 데이터", " : \n", scaled_iris_train_data)
print("테스트 데이터", " : \n", scaled_iris_test_data)

