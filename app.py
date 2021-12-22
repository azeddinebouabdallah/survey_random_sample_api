from flask import Flask, request
from flask_cors import CORS, cross_origin
import pandas as pd
app = Flask(__name__)
cors = CORS(app)

app.config['CORS_HEADERS'] = 'Content-Type'


import numpy as np
import random
from scipy.stats import gamma
import pickle5 as pickle
import json


def weights(v):
    #tmp=[1/x for x in v]
    
    tmp=[gamma.pdf(x+1, 2.7, loc=0, scale=1) for x in v]
    tmp2=[x/sum(tmp) for x in tmp]
    tmp3=[]
    for i in range(len(tmp2)):
        tmp3.append(sum(tmp2[0:i+1]))

    return tmp3



def sample(w):
    #draw sample
    r=random.random()
    tmp4=np.array(w)-r
    return np.where (tmp4>0)[0][0]



@app.route("/")
@cross_origin()
def helloWorld():
  return "Hello, cross-origin-world!"

@app.route("/reset")
@cross_origin()
def reset():
    with open('visits.pkl', 'rb') as f:
        visits = pickle.load(f)
    visits=np.zeros(24192)
    
    with open('visits.pkl', 'wb') as f:
        pickle.dump(visits, f)

    return "reset"


@app.route("/weights")
@cross_origin()
def get_weights():
    with open('visits.pkl', 'rb') as f:
        visits = pickle.load(f)

    w = weights(visits)

    # save w as a pickle file
    with open('weights.pkl', 'wb') as f:
        pickle.dump(w, f)
    return "Wow"




@app.route("/sample")
@cross_origin()
def get_sample():
    with open('weights.pkl', 'rb') as f:
        w = pickle.load(f)

    r = []

    for i in range(40):
        foo = int(sample(w))
        if not foo % 2 == 0:
            foo -= 1
        r.append(foo)


    out = {'visits': r}

    # save out as a json file
    with open('visits.json', 'w') as f:
        json.dump(out, f)

    # open visits pickle file
    with open('visits.pkl', 'rb') as f:
        visits = np.array(pickle.load(f))
        visits[r] +=1
        
    # save visits as a pickle file
    with open('visits.pkl', 'wb') as f:
        pickle.dump(visits, f)

    return json.dumps(out)

@app.route("/incrementp1")
@cross_origin()
def incrementP1():
    q1 = request.args.get('q1')
    q2 = request.args.get('q2')
    imId = request.args.get('imageId')
    df = pd.read_csv('results.csv')
    df2 = pd.read_csv('results2.csv')

    try:
        df2.iloc[imId, q1] += 1
        df2.iloc[imId, q2] += 1
    except:
        print("error")
    df.loc[int(q1),"Value"] += 1
    df.loc[int(q2),"Value"] += 1

    df.to_csv('results.csv', index=False)

    return "Done"
    
@app.route("/incrementp2")
@cross_origin()
def incrementP2():
    q1 = request.args.get('q1')
    q2 = request.args.get('q2')
    q3 = request.args.get('q3')
    imId = request.args.get('imageId')

    df = pd.read_csv('results.csv')
    df2 = pd.read_csv('results2.csv')

    try:
        df2.iloc[imId, q1] += 1
        df2.iloc[imId, q2] += 1
        df2.iloc[imId, q3] += 1
    except:
        print("error")

    df.loc[int(q1),"Value"] += 1
    df.loc[int(q2),"Value"] += 1
    df.loc[int(q3),"Value"] += 1

    df.to_csv('results.csv', index=False)

    return "Done"

@app.route("/useremail")
@cross_origin()
def userEmail():
    email = request.args.get('email')
    
    df = pd.read_csv("emails.csv")
    if (email == "anonymous"):
        email = request.remote_addr
    
    df = df.append([email])

    df.to_csv('emails.csv', index=False)

    return "Done"
    
if __name__ == "__main__":
  app.run(host='0.0.0.0', port=5000)