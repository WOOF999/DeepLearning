import numpy as np

# 입력 데이터 (키(cm), 몸무게(kg))
X = np.array([170, 175, 180, 165, 160, 185])
y = np.array([68, 75, 80, 55, 50, 90])

# 모델 파라미터 초기화 (w: 가중치, b: 절편)
w = 0.0
b = 0.0

# 학습률(learning rate) 설정
#기존에는 0.0001이었으나, 학습률이 너무 높아서 최적의 가중치로 수렴하지 못하고 발산하는 형태가 된다.
#따라서 학습률을 조정하여 비용함수가 최소가 되는 지점으로 수렴하도록 함. 
lr = 0.00001

# 예측 함수 정의
#train 함수 내부에서 전역변수 w,b의 값이 0에서 바뀌지 않는 것을 파악함
#따라서 w,b 변수로 넘겨주어 y_pred 값이 제대로 바뀌게 해야 함함
def predict(x,w,b):
    y_pred = w*x+b
    return y_pred

# 손실 함수 정의 (평균 제곱 오차)
def mse_loss(y_true, y_pred):
    mse = np.mean((y_true - y_pred)**2)
    return mse

# 경사 하강법(Gradient Descent)을 사용하여 손실 함수 최소화
def train(X, y, w, b, lr, epochs):
    for epoch in range(epochs):

        # y 예측 값 계산
        y_pred = predict(X,w,b)
        
        # 손실 함수 계산
        loss = mse_loss(y, y_pred)
        
        # 가중치(w)와 절편(b) 업데이트
        grad_w = np.mean((y_pred - y)*X)
        grad_b = np.mean(y_pred - y)
        w = w - lr*grad_w
        b = b - lr*grad_b
        
        # 로그 출력
        if epoch % 100 == 0:
            print("Epoch %d: loss=%.4f, w=%.4f, b=%.4f" % (epoch, loss, w, b))
    
        return w, b

# 학습 실행
w, b = train(X, y, w, b, lr, epochs=1000)
# 키가 159cm인 경우 몸무게 예측
# train 함수를 호출 한 후 리턴받은 w, b 변수를 파라미터로 넘겨주어 predict 값을 잘 받아오게 함

my_height = 159
my_weight = predict(my_height,w,b)

print("나의 키는 %dcm이고, 예측 몸무게는 %.1fkg입니다." % (my_height, my_weight))