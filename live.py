from feeds import fetch,run
from engine import Classifier,evaluate
from collections import Counter
import re

def acquire():
    return {'sources':[fetch('https://api.github.com/repos/scikit-learn/scikit-learn/issues?state=all&per_page=100&labels=Bug')]}


def acquire():
    urls=['https://api.github.com/repos/scikit-learn/scikit-learn/issues?state=all&per_page=100&labels='+label for label in ['Bug','Documentation','New%20Feature']]
    sources=[]
    for url in urls:
        source=fetch(url)
        # Retain only fields used by the model; exclude bodies and author details.
        source=dict(source,payload=[{'id':x['id'],'title':x['title'],'created_at':x['created_at'],
            'labels':[{'name':l['name']} for l in x['labels']],'html_url':x['html_url'],
            'is_pull_request':'pull_request' in x} for x in source['payload']])
        sources.append(source)
    return {'sources':sources}

def analyze(snapshot):
    mapping={'Bug':'bug','Documentation':'documentation','New Feature':'feature'}
    unique={}
    for source in snapshot['sources']:
        for issue in source['payload']:
            if issue.get('is_pull_request') or 'pull_request' in issue:continue
            labels={mapping[l['name']] for l in issue['labels'] if l['name'] in mapping}
            if len(labels)!=1:continue
            key=' '.join(re.findall(r'[a-z]+',issue['title'].lower()))
            if key:unique[key]=issue
    rows=sorted(unique.values(),key=lambda x:(x['created_at'],x['id']))
    split=int(len(rows)*0.8)
    train,test=rows[:split],rows[split:]
    def pairs(items):
        return [(x['title'],next(mapping[l['name']] for l in x['labels'] if l['name'] in mapping)) for x in items]
    training=pairs(train);testing=pairs(test)
    if len(set(label for _,label in training))<2 or not testing:
        raise ValueError('not enough labeled non-PR issues for a chronological train/test split')
    model=Classifier().fit(training)
    model.save('trained-model.json')
    return {'project':'TicketTriage','real_labeled_issues':len(rows),'train_examples':len(train),'test_examples':len(test),
            'train_class_counts':dict(Counter(label for _,label in training)),
            'test_class_counts':dict(Counter(label for _,label in testing)),
            'train_latest_created_at':train[-1]['created_at'],'test_earliest_created_at':test[0]['created_at'],
            'evaluation':evaluate(model,testing,threshold=0.45),
            'sample_predictions':[{'title':x['title'],'url':x['html_url'],'prediction':model.predict(x['title'],threshold=0.45)} for x in test[:5]],
            'note':'Maintainer labels are weak supervision; titles only, PRs excluded, duplicate normalized titles removed. Oldest 80% train, newest 20% test. Current labels can change; this is a snapshot benchmark, not a historical production backtest.'}


if __name__=='__main__': run(acquire,analyze)
