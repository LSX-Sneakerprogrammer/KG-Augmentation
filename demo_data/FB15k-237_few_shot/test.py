import numpy as np
import os

def get_relations(i):
    """Gets all labels (relations) in the knowledge graph."""
    # return list(self.labels)
    path = os.path.join('/export2/liushixuan/EvalKABERT/ANU/kg-bert-master/data/FB15k-237/spilt_train', f"relation{i}.tsv")
    with open(path, 'r') as f:
        cnt = 0
        for line in f:
            cnt += 1
    return cnt

dict = {}
for j in range(237):
    cnt = get_relations(j)
    dict[j] = cnt

x = sorted(dict.items(), key = lambda kv:(kv[1], kv[0]), reverse=True)
print(x[:20])