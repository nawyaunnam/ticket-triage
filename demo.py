import json
import tempfile
from pathlib import Path
from engine import Classifier, evaluate

train = [
    ('charged twice invoice payment refund','billing'),('billing payment card declined','billing'),
    ('refund subscription invoice charge','billing'),('cancel subscription payment','billing'),
    ('password login account locked','access'),('reset password sign in','access'),
    ('cannot login authentication account','access'),('login access password reset','access'),
    ('application crash error loading','bug'),('upload file error crash','bug'),
    ('page broken application fails','bug'),('error loading broken screen','bug')]
test = [('invoice refund please','billing'),('payment charged again','billing'),
        ('reset locked account','access'),('password authentication failed','access'),
        ('application upload crash','bug'),('broken page loading','bug')]
model = Classifier().fit(train)
with tempfile.TemporaryDirectory() as d:
    path = Path(d)/'model.json'
    model.save(path)
    loaded = Classifier.load(path)
    print(json.dumps({'evaluation':evaluate(loaded,test), 'prediction':loaded.predict('invoice payment refund'),
                      'unknown':loaded.predict('astronomy telescope constellation')}, indent=2))
