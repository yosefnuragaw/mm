from sentence_transformers import SentenceTransformer
from scipy.stats import beta
from sklearn.cluster import KMeans
from typing import List
import numpy as np 
import torch 
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from collections import defaultdict
import math
import heapq
from datetime import datetime

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

class ModelSelector:
    def __init__(self, models:List, args):
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2',device=DEVICE)
        self.k_means = KMeans(n_clusters= args.n_cluster)
        self.models = models
        self.r = args.sample_rate
        self.args = args
        
        if args.sample_rate > 1 :
            raise ValueError(f"Invalid sampling rate: '{args.sample_rate}'!")
    
    def _normalize_l2(self,data: List[float]):
        return normalize(data, norm='l2')

    def _sample(self, data: List):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Start Sampling")
        # Clustering K-Means (Cosine Distance)
        queries = [d.get("instruction") for d in data]
        norm = self._normalize_l2(self.encoder.encode(queries))
        self.k_means.fit(norm)
        cluster = self.k_means.labels_

        # Random sampling
        mask = np.zeros(len(cluster), dtype=int)
        for label in np.unique(cluster):
            indices = np.where(cluster == label)[0]
            n_samples = int(np.ceil(self.r * len(indices)))
            sampled = np.random.default_rng(seed=RANDOM_SEED).choice(
                indices, size=n_samples, replace=False
                )
            mask[sampled] = 1
        
        return [data for data, _ in zip(data, mask) if _ ]
         
    
    def cluster_sampling(self, data: List): #DONE (O(rKN)) 
        dsample = self._sample(data)
        res = list()

        #O(rKN)
        
        for m in self.models:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Start Evaluating {m}")
            eval = m.evaluate(dsample)
            res.append(eval.get("score"))

        print(f"[{datetime.now().strftime('%H:%M:%S')}] Cluster Sampling Done")
        return res

    def cluster_sampling_thompson(self, data: List): #DONE (O(N.p + N . (rK - p) + N) -> #O(N . rK))
        dsample = self._sample(data)
        res = list()
        dist = list()
        indices = np.random.choice(len(dsample), size=self.args.prior_size, replace=False)
        inverse_indices = np.setdiff1d(np.arange(len(dsample)), indices)

        dprior = [dsample[idx] for idx in indices]
        dinverse = [dsample[idx] for idx in inverse_indices]
        # O(N.p)
        idx = 1
        for m in self.models: 
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Start Computing prior {m}")
            ev = m.evaluate(dprior)
            dist.append((ev.get("correct")+1, ev.get("false")+1)) #Beta Initialization

        #O(N . (rK - p))
        for dp in dinverse: #O(rK - p)
            
            s = [beta.rvs(d[0], d[1], size=1) for d in dist] #O(N)
            idx = np.argmax(s)

            m = self.models[idx]
            ev = m.evaluate(dp)
            dist[idx] = (dist[idx][0]+ev.get("correct"), dist[idx][0]+ev.get("false"))

        # O(N) Not tracked yet in report [ ]
        for i, (a, b) in enumerate(dist):
            res.append(a / (a + b))
            
        return res

    def succesive_reject(self, data: List ): #DONE (O(N^2.b))
        if self.args.strategy not in ["vanilla","halving"] :
            raise ValueError(f"Invalid strategy, use ['vanilla','halving]!")

        dsample = self._sample(data)
        res = list()
        surv = [1 for _ in range(len(self.models))]
        t = 1

        # O(N^2.b)
        while sum(surv) > 1:
            indices = np.random.choice(len(dsample), size=self.args.eval_size, replace=False)
            deval = [data for data, _ in zip(dsample, indices) if _]

            st = list()
            for i, m in zip(surv,self.models):
                if i:
                    ev = m.evaluate(deval)
                    st.append(ev.get("score"))
                else:
                    st.append(np.nan) # Not survived in t-1

            res.append(st)
            
            if self.args.strategy == "vanilla":
                surv[np.nanargmin(st)] = 0
            elif self.args.strategy == "halving":
                temp_st = st.copy()
                for _ in range (sum(surv) // 2):
                    idx =np.nanargmin(temp_st) 
                    surv[idx] = 0
                    temp_st[idx] = np.nan
            else:
                raise ValueError(f"Unknown strategy: {self.args.strategy}. Expected 'vanilla' or 'halving'.")


        return res

    def ucb(self, data: List):#DONE O(T . N)
        dsample = self._sample(data)
        res = list()
        states = [(idx,0,0) for idx in range(len(self.models))]
        #O(T . (N+b))
        for t in range(1,self.args.rounds): #O(T)
            indices = np.random.choice(len(dsample), size=round(self.args.eval_rate*len(dsample)), replace=False)
            deval = [dsample[idx] for idx in indices]
            scores = list()
            for (_,n,q) in states: #O(N + b)
                scores.append(q+ self.args.coeff * math.sqrt(math.log(t)/ (n+1e-12)))
            
            idx = np.argmax(scores)
            ev = self.models[idx].evaluate(deval)
            nt = states[idx][1]+len(indices)
            qt = states[idx][2]+ ev.get("score")/ nt

            states[idx] = (idx,nt,qt)
            
        return heapq.nlargest(self.args.k_model, states, key=lambda x: x[2])

# if __name__ == "__main__":
#     sampler = ModelSelector(models= [])
#     l = [
#     "Sentence on is goff",                         
#     "This is sentence 2 speakng",                   
#     "The sun sets behind the mountains.",
#     "She enjoys reading historical fiction books.",
#     "Artificial intelligence is changing the world.",
#     "He forgot his umbrella on a rainy day.",
#     "They are planning a trip to Japan next spring.",
#     "Music can evoke strong emotions.",
#     "The stock market saw a sharp increase today.",
#     "A quick brown fox jumps over the lazy dog.",
#     "Learning Python has become very popular.",
#     "The children laughed as they played in the park.",
#     "He brewed a cup of coffee before his meeting.",
#     "Climate change is a pressing global issue.",
#     "The artist painted a beautiful landscape.",
#     "Technology evolves at an exponential rate."
#     ]
#     print(sampler.process(l))