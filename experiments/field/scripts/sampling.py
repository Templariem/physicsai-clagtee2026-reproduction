import collections,torch
def weights(rows):
    by=collections.defaultdict(lambda:collections.defaultdict(lambda:collections.Counter()))
    for r in rows:by[r['scene']['mode']][r['camera']][r['group']]+=1
    return torch.tensor([1/len(by)/len(by[r['scene']['mode']])/len(by[r['scene']['mode']][r['camera']])/by[r['scene']['mode']][r['camera']][r['group']] for r in rows],dtype=torch.float64)
