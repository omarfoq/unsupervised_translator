# Simple Non-Supervised Translator

Implementation of an unsupervised translator, we only use the word 
embeddings provided by fastText [[1]](https://arxiv.org/abs/1612.03651) provided [here](https://fasttext.cc/docs/en/crawl-vectors.html). 

## 1. Description of the system:

Let's consider a bilingual dictionary of size V_a (e.g French-English).

Let's define **X** and **Y** the **French** and **English** matrices.

They contain the embeddings associated to the words in the bilingual dictionary.

We want to find a **mapping W** that will project the source word space (e.g French) to the target word space (e.g English).

Procrustes : **W\* = argmin || W.X - Y ||  s.t  W^T.W = Id**
has a closed form solution:
**W = U.V^T  where  U.Sig.V^T = SVD(Y.X^T)**


## 2. Supported Languages
The system supports only English and French for now, other languages will be added in the future

## 3. Requirements
 * Scipy
 * Numpy
 
 Before getting the translator to work, you need to download 
 and unzip the English and French word representation in **data** directory. 
 You can get them from:
 
     https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.fr.vec
     https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.vec
## 4. Examples
To translate a list of words from English to French:

    python main.py -src "en" -trgt "fr" word1 word2 word3 ...
    
## References
[1] A. Joulin, E. Grave, P. Bojanowski, M. Douze, H. Jégou, T. Mikolov, FastText.zip: Compressing text classification models 

    @article{joulin2016fasttext,
    title={FastText.zip: Compressing text classification models},
    author={Joulin, Armand and Grave, Edouard and Bojanowski, Piotr and Douze, Matthijs and J{\'e}gou, H{\'e}rve and Mikolov, Tomas},
    journal={arXiv preprint arXiv:1612.03651},
    year={2016}
    }