# aind2-dl

### Instructions

1. Clone the repository and navigate to the downloaded folder.
	
	```	
		git clone https://github.com/udacity/aind2-dl.git
		cd aind2-dl
	```

2. Obtain the necessary Python packages, and switch Keras backend to Tensorflow.  
	
	For __Mac/OSX__:
	```
		conda env create -f requirements/aind-dl-mac.yml
		source activate aind-dl
		KERAS_BACKEND=tensorflow python -c "from keras import backend"
	```

	For __Windows__:
	```
		conda env create -f requirements/aind-dl-windows.yml
		activate aind-dl
		set KERAS_BACKEND=tensorflow
		python -c "from keras import backend"
	```

	For __Linux__:
	```
		conda env create -f requirements/aind-dl-linux.yml
		source activate aind-dl
		KERAS_BACKEND=tensorflow python -c "from keras import backend"
	```
	
3. Enjoy!

### IMDB movie review sentiment classification dataset

https://keras.io/api/datasets/imdb/

<p>Loads the <a href="https://ai.stanford.edu/~amaas/data/sentiment/">IMDB dataset</a>.</p>
<p>This is a dataset of 25,000 movies reviews from IMDB, labeled by sentiment
(positive/negative). Reviews have been preprocessed, and each review is
encoded as a list of word indexes (integers).
For convenience, words are indexed by overall frequency in the dataset,
so that for instance the integer "3" encodes the 3rd most frequent word in
the data. This allows for quick filtering operations such as:
"only consider the top 10,000 most
common words, but eliminate the top 20 most common words".</p>
<p>As a convention, "0" does not stand for a specific word, but instead is used
to encode any unknown word.</p>
<p><strong>Arguments</strong></p>
<ul>
<li><strong>path</strong>: where to cache the data (relative to <code>~/.keras/dataset</code>).</li>
<li><strong>num_words</strong>: integer or None. Words are
    ranked by how often they occur (in the training set) and only
    the <code>num_words</code> most frequent words are kept. Any less frequent word
    will appear as <code>oov_char</code> value in the sequence data. If None,
    all words are kept. Defaults to None, so all words are kept.</li>
<li><strong>skip_top</strong>: skip the top N most frequently occurring words
    (which may not be informative). These words will appear as
    <code>oov_char</code> value in the dataset. Defaults to 0, so no words are
    skipped.</li>
<li><strong>maxlen</strong>: int or None. Maximum sequence length.
    Any longer sequence will be truncated. Defaults to None, which
    means no truncation.</li>
<li><strong>seed</strong>: int. Seed for reproducible data shuffling.</li>
<li><strong>start_char</strong>: int. The start of a sequence will be marked with this
    character. Defaults to 1 because 0 is usually the padding character.</li>
<li><strong>oov_char</strong>: int. The out-of-vocabulary character.
    Words that were cut out because of the <code>num_words</code> or
    <code>skip_top</code> limits will be replaced with this character.</li>
<li><strong>index_from</strong>: int. Index actual words with this index and higher.</li>
<li><strong>**kwargs</strong>: Used for backwards compatibility.</li>
</ul>
<p><strong>Returns</strong></p>
<p>Tuple of Numpy arrays: <code>(x_train, y_train), (x_test, y_test)</code>.</p>
<p><strong>x_train, x_test</strong>: lists of sequences, which are lists of indexes
  (integers). If the num_words argument was specific, the maximum
  possible index value is <code>num_words - 1</code>. If the <code>maxlen</code> argument was
  specified, the largest possible sequence length is <code>maxlen</code>.</p>
<p><strong>y_train, y_test</strong>: lists of integer labels (1 or 0).</p>
<p><strong>Raises</strong></p>
<ul>
<li><strong>ValueError</strong>: in case <code>maxlen</code> is so low
    that no input sequence could be kept.</li>
</ul>
<p>Note that the 'out of vocabulary' character is only used for
words that were present in the training set but are not included
because they're not making the <code>num_words</code> cut here.
Words that were not seen in the training set but are in the test set
have simply been skipped.</p>
<hr />
