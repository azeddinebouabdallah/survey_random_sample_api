from apscheduler.schedulers.blocking import BlockingScheduler
import pickle5 as pickle
import numpy as np
from scipy.stats import gamma
import pickle5 as pickle

def weights(v):
    #tmp=[1/x for x in v]
    
    tmp=[gamma.pdf(x+1, 2.7, loc=0, scale=1) for x in v]
    tmp2=[x/sum(tmp) for x in tmp]
    tmp3=[]
    for i in range(len(tmp2)):
        tmp3.append(sum(tmp2[0:i+1]))

    return tmp3

def get_weights():
    with open('visits.pkl', 'rb') as f:
        visits = pickle.load(f)

    w = weights(visits)

    # save w as a pickle file
    with open('weights.pkl', 'wb') as f:
        pickle.dump(w, f)
    return "Weights updated"

scheduler = BlockingScheduler()
scheduler.add_job(get_weights, 'interval', hours=1)
scheduler.start()