### output files

- entity2id.txt : Triple, QA에 등장하는 모든 entity에 대해 id 부여
  - format : entity id count
- relation2id.txt : Triple, QA에 등장하는 모든 relation에 대해 id 부여
  - format : relation id count
- answer2id.txt : QA에 등장하는 모든 answer에 대해 id 부여
  - format : answer id count entityid
- word2id.txt : QA에 등장하는 모든 질문의 단어에 대해 id 부여
  - format : word id count
- KGE_train2id.txt : Triple id화
  - format : e1 e2 r (OpenKE 포맷)
- Fact_train2id.txt : QA의 Fact id화
  - format : e1 e2 r
- Question_train2id.txt : QA의 Question id화
- Answer_train2id.txt : QA의 Answer id화

- transe_xxx.ckpt : 학습된 KGE 모델
  - 1-1.txt, 1-n.txt, n-1.txt, n-n.txt, type_constrain.txt는 OpenKE 학습 파일
