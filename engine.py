"""Multinomial Naive Bayes with training-only vocabulary and safe JSON artifacts."""
import json
import math
import re
from collections import Counter
from pathlib import Path

def tokens(text):
    return re.findall(r'[a-z]+', text.lower())

class Classifier:
    def __init__(self, alpha=1.0):
        if not math.isfinite(alpha) or alpha <= 0: raise ValueError('alpha must be positive and finite')
        self.alpha = alpha
        self.classes, self.counts, self.vocabulary = Counter(), {}, set()
    def fit(self, examples):
        self.classes, self.counts, self.vocabulary = Counter(), {}, set()
        for text, label in examples:
            if not isinstance(label,str) or not label: raise ValueError('nonempty string labels required')
            terms = tokens(text)
            self.classes[label] += 1
            self.counts.setdefault(label, Counter()).update(terms)
            self.vocabulary.update(terms)
        if len(self.classes) < 2 or not self.vocabulary: raise ValueError('at least two classes and nonempty vocabulary required')
        return self
    def predict(self, text, threshold=0.6):
        if not self.classes: raise ValueError('fit the model first')
        if not 0 <= threshold <= 1: raise ValueError('threshold must be in [0,1]')
        known = Counter(t for t in tokens(text) if t in self.vocabulary)
        scores = {}
        for label, n in self.classes.items():
            denominator = sum(self.counts[label].values()) + self.alpha*len(self.vocabulary)
            score = math.log(n/sum(self.classes.values()))
            score += sum(freq*math.log((self.counts[label][term]+self.alpha)/denominator) for term,freq in known.items())
            scores[label] = score
        peak = max(scores.values())
        weights = {label:math.exp(score-peak) for label,score in scores.items()}
        total = sum(weights.values())
        probabilities = {label:value/total for label,value in weights.items()}
        best = sorted(probabilities,key=lambda label:(-probabilities[label],label))[0]
        accepted = bool(known) and probabilities[best] >= threshold
        return {'label': best if accepted else None, 'candidate':best, 'confidence':probabilities[best],
                'probabilities':probabilities, 'known_terms': sorted(known), 'abstained':not accepted}
    def save(self, path):
        if not self.classes: raise ValueError('fit the model first')
        Path(path).write_text(json.dumps({'format':1,'alpha':self.alpha,'classes':dict(self.classes),
            'counts':{label:dict(count) for label,count in self.counts.items()},'vocabulary':sorted(self.vocabulary)},sort_keys=True))
    @classmethod
    def load(cls,path):
        data = json.loads(Path(path).read_text())
        if data['format'] != 1: raise ValueError('unsupported model format')
        model = cls(data['alpha'])
        model.classes = Counter(data['classes'])
        model.counts = {label:Counter(count) for label,count in data['counts'].items()}
        model.vocabulary = set(data['vocabulary'])
        return model

def evaluate(model, examples, threshold=0.6):
    examples = list(examples)
    if not examples: raise ValueError('test examples required')
    predictions = [model.predict(text,threshold)['label'] for text,_ in examples]
    truth = [label for _,label in examples]
    labels = sorted(set(truth) | set(model.classes))
    per_class = {}
    for label in labels:
        tp = sum(p==label and y==label for p,y in zip(predictions,truth))
        fp = sum(p==label and y!=label for p,y in zip(predictions,truth))
        fn = sum(p!=label and y==label for p,y in zip(predictions,truth))
        per_class[label] = {'precision':tp/(tp+fp) if tp+fp else 0,
                            'recall':tp/(tp+fn) if tp+fn else 0,
                            'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0}
    majority = sorted(model.classes,key=lambda x:(-model.classes[x],x))[0]
    confusion = Counter(f'{y} -> {p if p is not None else "ABSTAIN"}' for p,y in zip(predictions,truth))
    return {'examples':len(truth),'accuracy':sum(p==y for p,y in zip(predictions,truth))/len(truth),
            'macro_f1':sum(x['f1'] for x in per_class.values())/len(labels),
            'coverage':sum(p is not None for p in predictions)/len(truth),
            'majority_baseline_accuracy':sum(y==majority for y in truth)/len(truth),
            'per_class':per_class,'confusion':dict(confusion)}
