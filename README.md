# KVQA

Knowledge Graph를 이용한 관광 데이터 VQA

## Requirements

- torch
- [OpenKE](https://github.com/thunlp/OpenKE)
  - KGE를 만들기 위함. 다른 API 사용 무방
- [OKT](https://konlpy.org/ko/latest/index.html)
  - 형태소 분석.
- OpenJDK 1.8

## 1. Prepare Data

서버 1 /workspace/datasets/KVQA 에 있는 raw data를 이용하여 실험하였습니다.

아래의 코드를 통해 필요한 파일을 생성합니다.

```
python data/prepare_data.py --kge_dim 300
```

- TransE 모델을 사용. Default Dim: 300

- [output file 설명](./data/README.md)

### TODO

현재 raw file을 먼저 읽게 되어 있음. 중간 파일을 저장하는 구문을 만들었지만, raw file에서의 예외들이 있어서, 정상동작되지 않음.

## 2. Train

```
./train.sh
```

- configs.py 내 파라미터 확인
  - self.KVQA.question_max_length = prepare_data.py로 부터 구해지는 질문 단어 개수 + n
  - self.KVQA.num_entity = data/KVQA/entity2id.txt 의 카운트
  - self.KVQA.num_relation = data/KVQA/relation2id.txt 의 카운트
- Default Model
  - Fusion model : MLP
  - Answer model
    - Answer space : CLS
    - Relation space : CLS

## 3. Evaluation

```
python eval.py
```

- Accuracy(All): 전체 answer candidate 중에서 Top 1이 정답인 경우
- Accuracy(5 choices): 5지선다, (정답은 무조건 0번, 나머지 겹치지 않게 random selected)
