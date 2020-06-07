from memm import *
from model1 import Model1
from model2 import Model2
from pathlib import Path
from itertools import chain, product
from heapq import nsmallest
from time import time


def accuracy(y_true, y_pred) -> float:
    return sum(y1 == y2 for y1, y2 in zip(y_true, y_pred)) / len(y_true)


def confusion_matrix(y_true, y_pred, y_range, n=None) -> Dict[Tuple[Tag, Tag], float]:
    cm = {(y1, y2): 0 for y1, y2 in product(y_range, repeat=2)}
    for y1, y2 in zip(y_true, y_pred):
        cm[y1, y2] += 1
    for y1 in y_range:
        s = sum(cm[y1, y2] for y2 in y_range)
        if not s:
            continue
        for y2 in y_range:
            cm[y1, y2] /= s
    if n:
        n_worst = nsmallest(n, y_range, key=lambda y: cm[y, y] if cm[y, y] != 0 else 1)
        cm = {(y1, y2): cm[y1, y2] for y1, y2 in cm if y1 in n_worst or y2 in n_worst}
    return cm


if __name__ == '__main__':
    total = time()

    # Model 1
    model1_path = Path('model/model1')
    model1 = None
    try:
        model1 = load_model(model1_path)
    except FileNotFoundError:
        train1 = preprocess(Path('data/train1.wtag'))
        model1 = Model1(lambda_=1, beam=3)
        model1.fit(*train1)
        model1.save_model(model1_path)
        train1_tags = list(chain(*train1[1]))
        print(f'Model 1 Train Accuracy: {accuracy(train1_tags, train1_tags)}')
    finally:
        test1 = preprocess(Path('data/test1.wtag'))
        test1_true_tags = list(chain(*test1[1]))
        test1_pred_tags = list(chain(*model1.predict(test1[0])))
        print(f'Model 1 Test Accuracy: {accuracy(test1_true_tags, test1_pred_tags)}')
        confusion = confusion_matrix(test1_true_tags, test1_pred_tags, model1.tag_vocabulary, n=10)
        tags = [tag for tag in model1.tag_vocabulary if (tag, tag) in confusion]
        tags.sort(key=lambda tag: confusion[tag, tag])
        print(f'Model 1 Test Confusion Matrix:')
        print(''.ljust(5) + ''.join(t.ljust(5) for t in tags))
        rows = [r.ljust(5) + ''.join([f'{confusion[r, c]:.2f}'.ljust(5) for c in tags]) for r in tags]
        print('\n'.join(rows))

    # Model 2
    model2_path = Path('model/model2')
    try:
        model2 = load_model(model2_path)
    except FileNotFoundError:
        train2 = preprocess(Path('data/train2.wtag'))
        model2 = Model2(lambda_=0.1, beam=3)
        model2.fit(*train2)
        model2.save_model(model2_path)
        train2_tags = list(chain(*train2[1]))
        print(f'Model 2 Train Accuracy: {accuracy(train2_tags, train2_tags)}')

    print(f'Total runtime: {time() - total:.2f}s')
