# 2021-1_ML_Project

I. 문제 정의

![image](https://user-images.githubusercontent.com/86652565/148222261-af3d58e9-9cdb-4896-835d-c888d9cd9fe5.png)

통칭 심장병이라고 불리는 심근경색은 사망률 원인 2위에 달할 만큼 예전부터 큰 인류의 이슈였다. 특히나 다른 부위도 아닌 심장인 만큼 전문가들은 증상이 나타나면 병원에 도착해 치료를 받기까지 120분내로 권장하고 있다. 그 이상 넘어가면 생명과도 직결되는 문제이기 때문이다. 또한 기사와 같이 심근경색은 가족력이 있는 경우와 심장 과부하를 일으키는 고혈압, 혈중 콜레스트롤, 또한 생활 스트레스가 높은 경우 급성심근경색증의 위험이 높다고 알려져 있다. 따라서 기존 심근경색 환자들의 데이터를 분석하여 본인의 데이터로 심장병 발병 가능성의 유무를 대략적으로 판단해주어, 해당되는 사람들에게 경각심을 고취시킬 수 있는 모델을 만드는 것이 본 프로젝트의 목표이다. 



II. 데이터셋 설명

![image](https://user-images.githubusercontent.com/86652565/148223333-7e516607-e346-4de0-b073-af9eb54463b2.png)


데이터 셋은 위 그림과 같다. 각 컬럼에 대한 설명은 다음과 같다.
- age : 나이
- sex : 성별 ( 남자 = 1, 여자 = 0 )
- cp : 가슴 통증 종류 
( 무증상 = 0, 전형적인 협심증 = 1, 전형적이지 않은 협심증 = 2, 협심증이 아닌 통증 = 3 )
- trestbps : 안정혈압 ( mmHg )
- chol : 콜레스트롤 수치 ( mg/dl )
- fbs : 공복혈당 ( 공복혈당 > 120mg/dl = 1, 그렇지 않으면 = 0 )
- restecg : 심전도 결과
- thalach : 최대 심박수
- exang : 운동 유발성 협심증 ( 해당 = 1, 해당하지 않음 = 0 )
- oldpeak : 휴식으로 유발된 ST우울증
- slope : 가장 높은 ST분절의 경사
- ca : 주요 혈관 수
- target : 병 보유 유무 ( 보유 = 1, 미보유 = 0 )


III. 알고리즘 채택

0. 데이터 전처리
- info()와 isnull(), duplicated() 등을 통해 결측값과 중복값이 없음을 알 수 있다.
- thal 열은 관련 자료 부족으로 drop()을 이용해 삭제한다.
- cp, slope 열은 categorical data이기 때문에 dummy variable로 변환한다. 
(get_dummies() 함수 이용, 기존의 cp, slope열은 drop()을 이용해 삭제)

![image](https://user-images.githubusercontent.com/86652565/148223426-a088f76b-b935-4d86-a309-08435cd34f81.png)

- target열이 실제 병 발병 유무이므로 y에 저장한다.
- x_data에 target열을 삭제한 나머지를 저장한다.
- 이상치 제거 : matplotlib의 boxplot()을 활용해 연속 데이터들의 이상치를 확인한다.

![image](https://user-images.githubusercontent.com/86652565/148222399-15de33a7-4777-46e4-ba70-3004762dbb30.png)


확인 결과 위와 같이 눈에 띄는 이상치들이 존재한다. 이상치 처리를 위해 빨간색 원 안에 있는 데이터들은 그 행을 통째로 지워준다.

※ matplotlib의 boxplot()은 데이터들의 이상치를 확인할 수 있는 메소드이다. boxplot은 사분위수를 이용하여 데이터를 나타내는데, 가운데 사각형의 노란 직선이 중위수(median)이며, 사각형의 윗변과 밑변이 각각 3분위수(Q3), 1분위수(Q1)이다. 그리고 사각형 외부로 뻗어있는 직선을 울타리라고 부르는데, 각 직선의 끝점은 다음과 같이 계산한다.

![image](https://user-images.githubusercontent.com/86652565/148223462-c04c6a20-01b3-4ae1-b082-ab1817b58a8d.png)

보통 이 울타리 밖의 data들을 이상치(outlier)라고 한다. 다만 위 그림에서 울타리 밖의 데이터를 전부 제거하기에는 데이터 수가 많이 줄어드므로, 눈에 확 띄는 부분만 제거하기로 한다.

- 최소-최대 정규화
data를 사용할 때는 각 데이터별 특성에 따라 특정 부분이 비정상적으로 전체 결과에 크게 영향을 끼칠 수 있기 때문에 정규화가 필요하다. 보통 정규화에는 두 가지 방법이 존재한다. 최소-최대 정규화(Min-Max Normalization)과 Z-점수 정규화(Z-Score Normalization)이다. 최소-최대 정규화는 각 컬럼들의 최댓값과 최솟값을 계산하여 그 안에 데이터들이 일정한 척도로 존재하게 하는 것으로, 이상치의 영향을 많이 받는다. 이러한 단점을 보완하기 위한 것이 Z-점수 정규화인데, Z-점수 정규화는 각 컬럼들의 평균과 표준편차를 이용한 것으로, 최소-최대 정규화의 단점을 보완하였다고 이야기할 수 있다. 그러나 이 프로젝트에서는 위의 방법으로 눈에 띄는 이상치를 제거하였으므로, 가장 널리 사용되는 최소-최대 정규화를 사용하기로 한다. 아래 그림은 최대-최소 정규화를 마친 데이터이다.

![image](https://user-images.githubusercontent.com/86652565/148223651-484003e0-733c-4fb3-b9ce-becf35705788.png)


- train data와 test data 분류
train_test_split() 함수로 train data는 80%, test data는 20%로 설정한다.
※ 보통 데이터로 모델을 만들 때 갖고 있는 모든 데이터를 모델에 학습 시키지 않고, train set과 vaildation set으로 나누어 모델의 성능이 좋은지 검증한다. 이 프로젝트에서 test data는 위 설명에서의 vaildation set의 역할을 수행한다고 볼 수 있다.


1. KNN
- sklearn의 KNeighborsClassifer을 사용한다.
- 최적의 k를 찾기 위해서 k를 1~29까지 1씩 증가시켜 가면서 train data들을 fit함수에 적용시킨 후, test data와 score함수를 사용하여 정확도를 조사한다.
- 밑의 그림에서와 같이 가장 높은 score는 k=12일 때이고, 그때의 정확도는 81.67% 임을 알 수 있다.

![image](https://user-images.githubusercontent.com/86652565/148223512-640d2954-e44a-4806-a11e-3162d5b77a7d.png)


2. Decision Tree
- sklearn의 DecisionTreeClassifier을 사용한다.
- decision tree는 항목 분리 기준 즉 불순도(impurity)의 측정에 따라 매개변수로 ‘gini’와 ‘entropy’ 등을 가질 수 있고, 항목 분리시 그 방법에 따라 ‘random’, ‘best’를 가질 수 있으며, 트리 모형의 최대 깊이인 max_depth에 따라 결과가 달라질 수 있으므로 각 경우를 실험해본다. 이 때 max_depth는 커질수록 과적합이 일어날 수 있으므로 주의해야 한다.

![image](https://user-images.githubusercontent.com/86652565/148222577-2dbfa152-10d7-401a-a034-ff5c8e39c56d.png)


max_depth = 5이고 splitter = ‘random’일 때, criterion에 관계없이 train 데이터를 fit함수에 적용시킨 후 와 test 데이터로 score함수를 사용하였을 때의 score가 81.67%로 가장 높았다.

3. Random Forest
- sklearn.ensemble 의 RandomForestClassifier을 이용한다.
- n_estimators는 랜덤 포레스트 안의 결정 트리 개수로, 기본적으로 클수록 각 tree의 정확도가 증가한다. 그러나 각 결정 트리들 간의 correlation이 증가하기 때문에 꼭 좋기만 하지는 않을 수 있다. 다음은 n_estimators들을 적절히 변화시켜가면서 위의 다른 모델들과 같이 fit함수와 score함수를 사용해 보았다.

![image](https://user-images.githubusercontent.com/86652565/148222651-6757451e-fa02-4275-913c-6f6184dca16e.png)


위 결과에 따라 무조건 tree개수가 많은 것이 좋지 않음을 알 수 있었다. 특히나 위 두 분류에서 score의 최고값이 81.67%로 같은 상황에서 score을 조금 더 위로 올릴 수 있었다. n_estimators = 200으로 정하자.

- 또 다른 파라미터 max_features는 분할을 할 때 고려할 feature의 개수이다. 위 데이터셋은 feature(column)이 17개이므로, 1개부터 17개 중 몇 개의 feature을 고려하면 좋을지 알아보았다.

![image](https://user-images.githubusercontent.com/86652565/148222695-89733648-119a-484c-bc3a-9331bef2281f.png)



3개 또는 4개의 feature을 트리에 포함시켰을 때 가장 높은 score을 얻을 수 있었다.

4. Support Vector Machine
- sklearn의 SVC를 이용한다.
- SVM에서는 Kernel을 조정할 수 있다. 선형으로 decision boundary를 깔끔하게 계산하기 어려운 경우, 데이터들을 고차원으로 보내서 보다 깔끔한 decision boundary를 얻기 위함인데, 이때 사용되는 것이 Kernel함수이다. 데이터들의 차원을 높이는 것은 실제로 매우 복잡하나, Kernel함수는 벡터들의 내적만 계산할 수 있게 해준다. Kernel함수를 linear(선형), rbf, poly, sigmoid로 조정해보자.

![image](https://user-images.githubusercontent.com/86652565/148222734-0d3a3bd4-3d2b-4f2b-99ca-5ebbc589e96a.png)

Kernel 함수를 rbf로 지정하였을 때 가장 score가 높음을 알 수 있다.


5. K-fold 교차 검증
- 위에서 train_test_split() 함수로 train data는 80%, test data는 20%로 설정하였는데, 사실 이것만으로는 부족할 수 있다. 왜냐하면 데이터가 충분히 많지 않은 경우에 아래 그림과 같이 어디 부분이 train data로 되었는지에 따라 결과가 달라질 수 있기 때문이다. 따라서 위의 모델들을 K-fold 교차 검증을 이용하고자 한다. k=5로 한다면, 다섯 번의 학습과 검증을 거쳐 나온 score의 평균을 계산하여 과적합의 가능성을 줄이는 방법이다.

![image](https://user-images.githubusercontent.com/86652565/148223784-86ab4107-daad-4fed-8131-06f55277dca3.png)


![image](https://user-images.githubusercontent.com/86652565/148222795-e5f8cebd-4569-4d6a-84c2-2f8b96ec1cf0.png)


- Ramdom Forest로 앙상블 한 모델의 K-fold 교차 검증까지 마쳤을 때의 score가 가장 높으므로 가장 적합한 모델이라고 할 수 있다.


IV. 결과 분석

1. 중요도 분석
원 데이터 셋에서도 봤듯, 심장병에 영향을 줄 수 있는 특성은 매우 다양하다. 물론 꾸준히 정기검진을 받아 건강 관리하는 것이 가장 바람직하겠으나, Random Forest를 이용하여   17개의 특성 중 심장병과 관련하여 중요하게 생각해야 할 특성을 골라보자. RandomForestClassifer의 feature_importances_ 메소드를 사용하여 각 컬럼이 target에 얼마나 영향을 끼치는지 알아보았다.

![image](https://user-images.githubusercontent.com/86652565/148222841-16382b9e-2820-4484-9136-1d36afe91ae3.png)


위 표를 보면 눈에 띄게 중요한 것이 thalach, oldpeak, ca, cp 정도가 되겠다. 우선 분류형이 아닌 데이터부터 각 특성별로 target과의 연관성을 정리해보자.

![image](https://user-images.githubusercontent.com/86652565/148222914-a0b11dc1-c070-4edd-95db-e9cf3ff591d0.png)

우선 thalach, 즉 최대 심박수는 높을수록 심장병의 위험이 높다고 할 수 있겠다. oldpeak(ST 우울증)은 없을수록 심장병의 위험이 높고, ca(형광투시법으로 보이는 주요 혈관의 수)도 없을수록 심장병이 위험이 높다고 할 수 있겠다. 다음은 분류형 데이터인 cp의 분석이다. 오른쪽 그림과 같이 무증상(cp = 0)인 경우보다 어떠한 방식으로도 통증이 있는 경우 (cp = 1, 2, 3)가 심장병의 위험이 훨씬 높다고 할 수 있다.

![image](https://user-images.githubusercontent.com/86652565/148222926-0443ac65-7b25-4de2-9f75-5bbc76a07d0f.png)



2. 결론
위 데이터셋의 17개의 특성 중 통상적으로 높은 관련성을 보이리라고 예상되는 특성은 나이, 가슴 통증, 혈압, 콜레스트롤 등이 있을 것이다. 보통 나이가 많을수록, (안정)혈압과 콜레스트롤이 높을수록 심장병의 위험하다고 생각할 것이다. 이제 마지막으로 위 3개의 특징을 분석해보자. 

![image](https://user-images.githubusercontent.com/86652565/148222963-5570a6fc-678f-460b-a0dd-d1c2eacc2fad.png)

실제로 각각 평균이 위 중요도 분석에서의 특성들보다는 큰 차이를 보이지 않음을 알 수 있었다. 위 중요도 분석을 통해 age, trestbps, chol의 중요도는 각각 8.87, 7.62, 8.40이었다. 물론 결코 작지 않은 중요도이나, 위 데이터셋을 통해 다른 요인도 중요할 수 있다는 것을 알 수 있었다.

3. 한계점
우선 이 데이터셋이 만들어진지 오래되었고, 300명이라는 적은 숫자의 사람들을 대상으로 한 데이터셋이라 현실과는 조금 다를 수 있다. 또한 본 분석은 비선형 분류 위주로 분석을 진행하였다. 선형 모델은 학습속도와 예측속도가 빠르다는 장점이 있으나, 종종 계수의 값들을 설명하기 어려울 수 있다고 판단하였기 때문이다. 또한 본 데이터는 크기가 크지 않기에, KNN과 Decision Tree와 같은 모델은 이해하고 설명하기 쉽다는 장점이 크게 작용할 수 있으리라고 생각하였다.
앞서 언급하였듯 데이터 크기가 크지 않기에 위의 중요도 분석으로 나온 특성들이 심장병을 유발하는 대표요인이라고 확정짓기 어려울 수 있다. 예를 들면 특성 중 oldpeak과 target의 관계를 명확히 설명하지 못했다. 그러나 심장병은 평소에 그 전조증상이 있더라도 잘못 판단되기 쉬우며, 그에 비해 생명에 치명적이기 때문에, 적어도 위 데이터분석으로 자신의 데이터를 측정하여 심장병의 위험이 있다고 판단되면 정밀검사를 받는 것이 도움이 되겠다.


V. 참고문헌

심장병 위험 : https://jhealthmedia.joins.com/article/article_view.asp?pno=22840

심장병 원인 : http://www.ikunkang.com/news/articleView.html?idxno=20078

데이터 출처 : https://www.kaggle.com/ronitf/heart-disease-uci

이상값 확인 : https://rfriend.tistory.com/410

데이터 정규화 : http://hleecaster.com/ml-normalization-concept/

K-fold 검증 :
https://blog.naver.com/PostView.nhn?blogId=winddori2002&logNo=221667083964

알고리즘 채택 :
https://www.kaggle.com/cdabakoglu/heart-disease-classifications-machine-learning

중요도 예측 : https://bizzengine.tistory.com/182


