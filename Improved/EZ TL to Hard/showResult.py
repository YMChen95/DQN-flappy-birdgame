import pylab as pl
import pandas as pd
import numpy as np
import pdb
def smooth(x,kernal):
    k_size = len(kernal)
    # pdb.set_trace()
    x = np.concatenate([np.tile(x[0],k_size-1),x,np.tile(np.mean(x[-k_size:]),k_size-1)])
    out = []
    for i in range(x.shape[0]-k_size):
        out.append(np.dot(x[i:i+k_size],kernal))
    return out



log = pd.read_csv('train_copy.log',header=None)

pl.figure()
pl.plot(log[0],label='reward')
pl.xlabel('Trained Iter')
pl.ylabel('reward')
pl.legend()

kernal = np.ones(51)/51.0
reward_s = smooth(log[0].values,kernal )

pl.figure()
pl.plot(reward_s,label='reward')
pl.xlabel('Trained Iter')
pl.ylabel('reward')
pl.legend()




pl.figure()
pl.plot(log[1],label='game score')
pl.xlabel('Trained Iter')
pl.ylabel('score')
pl.legend()

score_s = smooth(log[1].values,kernal )
pl.figure()
pl.plot(score_s,label='game score')
pl.xlabel('Trained Iter')
pl.ylabel('score')
pl.legend()


pl.show()
