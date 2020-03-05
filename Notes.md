#### Program structure

1. create `corpus.txt` file with space delimeted sentences which are delimited by newlines, using `prepare.py`

2. create pickled data files that are needed for training using `preprocess.py`.
   
   - params: `--ngrams --unk "|" --rand_window`

3. run `train.py` to traing the word embeddings
   
   - params: `--ngrams --n_negs 5 --weights --ss_t "1e-5"`
   
   - use additional params for batch size (`--mb`), epochs (`--epoch`)  as you like

Run without `--ngrams` to train reference model.

**What does `preprocess.py` do?**

- `convert` method generates a `skipgram` for each word:
  
  - `owords` is the _context_ from left to right
    
    - padded with `<|>` before and after if not present, since the fixed `window` is taken into account.

- this is then saved into binaries.

**What are the `iwords` and `owords`**?

loaded pairs from `PermutatedSubsampledCorpus` -> already like that from file.

- `iword`: current word $w_t$

- `owords:`: context words $w_c$ with $c \in \mathcal{C}_t$ 

#### Required adjustments:

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

**Still todo (maybe)**

- _„When building the word dictionary, we keep the words that appear at least 5 times in the training set“_ 
  
  - need to maybe reduce that threshold. Set the words to _unknown_ (`<|>`)? It might be smart to use a different character, though.

- _„we solve our optimization problem by performing SGD“_ *„we use a linear decay on the step size”*
  
  - currently using Adam. 

- use logistic loss function $l:x ↦ \log(1+e^{-x})$ instead of LogSigmoid!

- do not compute gradient on the unknown char!
  - Actually, this is already implemented. Passing a `padding_idx` to an `nn.Embedding` has exactly that effect.
  

**just a parameter**

- `ss_t`: `1e-4` (rejection threshold)

- `n_negs`: `5`

**First impressions** 
Takes a very long time to train. One iteration needs approximately 25 seconds vs 9-10 iterations/sec
when using the reference implementation. Will try to optimise a bit by sorting the corpus.

Unfortunately, this didn't help much. In fact's it's not speeding up the training at all!

FIXED - problem was a for-loop over the forward passes to the embeddings layer. Building the Index-Tensor for the whole batch instead made the training faster.

NOTE: It's important that the first entry of index 0 is the unknown/ nonexistent character!!! 

- And it shouldn't be trained...

- And the vector should be initialised as zero-vector...
