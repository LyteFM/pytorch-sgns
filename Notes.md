**What does `preprocess.py` do?**

- `convert` method generates a `skipgram` for each word:
  
  - `owords` is the _context_ from left to right
    
    - padded with `<UNK>` before and after if not present, since the fixed `window` is taken into account.

- this is then saved into binaries.

**What are the `iwords` and `owords`**?

loaded pairs from `PermutatedSubsampledCorpus` -> already like that from file.

- `iword`: current word $w_t$

- `owords:`: context words $w_c$ with $c \in \mathcal{C}_t$ 

#### Required adjustment:

**some effort**

- `window` -> must sample uniformly between 1 and 5 instead of just using a fixed size
  
  - I'll keep it fixed, else batching becomes unnecessarily difficult... usign a window of size 3 instead.

- _„probability proportional to the square root of the uni-gram frequency”_
  
  - can specify `weights` when initialising SGNS! adjust the function there.
    - done.

- Implementation of the `n-grams` and the adjusted scoring function:
  
  - **preprocessing**
    
    - Additional step to compute the n-grams and adding them to the dictionary; Hashing, building representation...
      - done, but not using hashing
  
  - **training**
    
    - instead of just retrieving for current word and context from `Word2Vec` during loss computation:
      
      - retrieve _all n-gram vectors_ and context
      
      - compute loss based on those

**little effort, might be OK to do differently**

- _„When building the word dictionary, we keep the words that appear at least 5 times in the training set“_ 
  
  - need to maybe reduce that threshold and set other words to _unknown_

- _„we solve our optimization problem by performing SGD“_
  
  - currently using Adam.

- _„we use a linear decay on the step size”_

- use logistic loss function $l:x ↦ \log(1+e^{-x})$ instead of LogSigmoid

**just a parameter**

- `ss_t`: `1e-4` (rejection threshold)

- `n_negs`: `5`


** First impressions** 
Takes a very long time to train. One iteration needs approximately 25 seconds vs 9-10 iterations/sec
when using the reference implementation. Will try to optimise a bit by sorting the corpus.

Unfortunately, this didn't help much... seems to be due more backtracking?
Or rather: due to looping over the forwarding...

FIXED - problem was a for-loop over the forward passes to the embeddings layer.
NOTE: It's important that the first entry of index 0 is the unknown/ nonexistent character!!!
