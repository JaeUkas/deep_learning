# 입력 필드들과 출력 필드가 같은 파일 내에 제공될 경우, Pandas DataFrame의 drop 메서드와 Pandas Series의 copy 메서드를 이용하여 분리
import pandas as pd

my_data = pd.read_csv("datasets/iris/iris.csv")

my_data_label = my_data['class'].copy()
# 원 DataFrame 객체에서 인덱서를 사용하여 ‘class’ 열만 선택한 후 이를 복사 (출력)
my_data_inputs = my_data.drop('class', axis = 1)
# 원 DataFrame 객체에서 drop 메서드를 이용하여 ‘class’ 열만 제거된 새로운 DataFrame 객체를 얻음 (입력)


