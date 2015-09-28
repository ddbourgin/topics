#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt

def clean_texts(texts, tokens, misc_stops=None):
    """
    Remove stopwords and punctuation from a collection of texts
    """
    import string
    from nltk.corpus import stopwords
    stops = set(stopwords.words('english'))
    stops = stops.union(set(string.punctuation))
    if misc_stops:
        stops = stops.union(set(misc_stops))
    stop_idxs = np.concatenate([np.argwhere(tokens == i).flatten() for i in stops])
    cleaned_texts = []
    for doc in texts:
        bad_idxs = [np.argwhere(doc == i).flatten() for i in stop_idxs]
        doc = np.delete(doc, np.concatenate(bad_idxs))
        cleaned_texts.append(doc)
    return np.asarray(cleaned_texts)


def lemmatize_n_tokenize(texts, tokens):
    """
    Lemmatize and stem all word tokens in a collection of texts
    """
    from nltk.stem.wordnet import WordNetLemmatizer
    from nltk.stem.snowball import SnowballStemmer
    lmtzr = WordNetLemmatizer()
    stemmer = SnowballStemmer("english")
    lemma_n = [lmtzr.lemmatize(i, 'v') for i in tokens]
    lemma_n_stem = [stemmer.stem(i) for i in lemma_n]
    simpl_tokens = np.unique(lemma_n_stem)
    token_translator = np.concatenate(
            [np.argwhere(simpl_tokens == i).flatten() for i in lemma_n_stem])
    simpl_texts = []
    for doc in texts:
        for idx,word in enumerate(doc):
            doc[idx] = token_translator[word]
        simpl_texts.append(doc)
    return np.asarray(simpl_texts), simpl_tokens


class TopicModel(object):

    def __init__(self, T, **kwargs):
        self.n_topics = T

        if 'alpha' in kwargs.keys():
            self.alpha = (kwargs['alpha']) * np.ones(self.n_topics)
        else:
            self.alpha = (50./self.n_topics) * np.ones(self.n_topics)

        if 'beta' in kwargs.keys():
            self.beta = kwargs['beta']
        else:
            self.beta = 0.01


    def get_text_info(self, processed_texts, tokens=[]):
        if list(tokens):
            self.tokens = tokens
        self.n_documents = len(processed_texts)
        self.n_tokens = len(np.unique(self.tokens))
        self.n_words = np.sum(np.array([len(doc) for doc in processed_texts]))
        self.word_document = np.zeros(self.n_words)

        # now that we know the number of tokens in our corpus, we can set beta
        self.beta = self.beta * np.ones(self.n_tokens)

        count = 0
        for doc_idx,doc in enumerate(processed_texts):
            for word_idx, word in enumerate(doc):
                word_idx = word_idx + count
                self.word_document[word_idx] = doc_idx
            count = count + len(doc)


    def preprocess_texts(self, texts):
        processed_texts, self.tokens = word_tokenize(texts)
        self.get_text_info(processed_texts)
        return processed_texts


    def train(self, processed_texts, n_gibbs=2000):
        """
        Trains a topic model on the documents in texts. Assumes texts is
        an array of subarrays, where each subarray corresponds to a
        separate document.
        """
        C_wt, C_dt, topic_asgnmnts = self.gibbs_sampler(n_gibbs, processed_texts)
        self.estimate_model_params(C_wt, C_dt)
        self.plot_topic_assignments(topic_asgnmnts)
        return C_wt, C_dt, topic_asgnmnts


    def plot_topic_assignments(self, topic_asgnmnts):
        plt.matshow(topic_asgnmnts, interpolation='nearest', aspect='auto')
        plt.colorbar(ticks=np.arange(self.n_topics))
        plt.grid(False)
        plt.title('Topic Assignments')
        plt.xlabel('Gibbs Iteration')
        plt.ylabel('Word Index')
        plt.show()


    def what_did_you_learn(self, top_n=10):
        """
        Prints the top n most probable words for each topic
        """
        for tt in xrange(self.n_topics):
            top_idx = np.argsort(self.phi[:, tt])[::-1][:top_n]
            top_tokens = self.tokens[top_idx]
            print('\nTop Words for Topic %s:\n'%(str(tt)))
            for token in top_tokens:
                print('\t%s\n'%(str(token)))


    def estimate_model_params(self, C_wt, C_dt):
        print('Estimating model parameters')
        self.phi = np.zeros([self.n_tokens, self.n_topics])
        self.theta = np.zeros([self.n_documents, self.n_topics])

        for ii in xrange(self.n_tokens):
            for jj in xrange(self.n_topics):
               self.phi[ii, jj] = (C_wt[ii, jj] + self.beta[0]) /\
                    (np.sum(C_wt[:, jj]) + self.n_tokens * self.beta[0])

        for dd in xrange(self.n_documents):
            for jj in xrange(self.n_topics):
                self.theta[dd, jj] = (C_dt[dd, jj] + self.alpha[0]) /\
                    (np.sum(C_dt[dd, :]) + self.n_topics * self.alpha[0])


    def estimate_conditional(self, token_idx, doc_idx, C_wt, C_dt):
       # Eq. 1 in Topic Model notes
       p_vec = np.zeros(self.n_topics)
       for topic_idx in xrange(self.n_topics):
           frac1 = (C_wt[token_idx, topic_idx] + self.beta[0]) /\
                        (np.sum(C_wt[:, topic_idx]) + self.n_tokens * self.beta[0])
           frac2 = (C_dt[doc_idx, topic_idx] + self.alpha[0]) /\
                        (np.sum(C_dt[doc_idx, :]) + self.n_topics * self.alpha[0])
           p_vec[topic_idx] = frac1 * frac2
       return p_vec / np.sum(p_vec)


    def gibbs_sampler(self, n_gibbs, processed_texts):
        # Initialize count matrices
        C_wt = np.zeros([self.n_tokens, self.n_topics])
        C_dt = np.zeros([self.n_documents, self.n_topics])
        topic_asgnmnts = np.zeros([self.n_words, n_gibbs+1])

        # Randomly initialize topic assignments for words
        for ii in xrange(self.n_words):
            token_idx = np.concatenate(processed_texts)[ii]
            topic_asgnmnts[ii, 0] = np.random.randint(0, self.n_topics)

            doc = self.word_document[ii]
            C_wt[token_idx, topic_asgnmnts[ii, 0]] += 1
            C_dt[doc, topic_asgnmnts[ii, 0]] += 1

        for ii in xrange(n_gibbs):
            print('Running Gibbs iteration %s of %s'%(str(ii + 1), str(n_gibbs)))
            for jj in xrange(self.n_words):
                token_idx = np.concatenate(processed_texts)[jj]

                # Decrement count matrices by 1
                doc = self.word_document[jj]
                C_wt[token_idx, topic_asgnmnts[jj, ii]] -= 1
                C_dt[doc, topic_asgnmnts[jj, ii]] -= 1

                # Draw new topic from our approximation of the conditional dist.
                p_topics = self.estimate_conditional(token_idx, doc, C_wt, C_dt)
                sampld_topic = np.nonzero(np.random.multinomial(1, p_topics))[0][0]

                # Update count matrices
                C_wt[token_idx, sampld_topic] += 1
                C_dt[doc, sampld_topic] += 1
                topic_asgnmnts[jj, ii+1] = sampld_topic
        return C_wt, C_dt, topic_asgnmnts


if __name__ == '__main__':
    # run parameters
    n_topics = 5
    n_train_docs = 50
    n_gibbs  = 100

    # load data
    texts = np.load('./TASA_texts.npz')['texts']
    tokens = np.load('./TASA_tokens.npz')['tokens']

    # shuffle texts
    shuffle_idxs = np.arange(texts.shape[0])
    texts = texts[shuffle_idxs]
    texts = texts[:n_train_docs]

    # remove stopwords
    remove_tokens = ['NUMBER', '--', 'j', '$XXXX', "a'", '.and', '.the', '/or', '?.']
    processed_texts = clean_texts(texts, tokens, misc_stops=remove_tokens)
    processed_texts, tokens = lemmatize_n_tokenize(processed_texts, tokens)

    # train model
    myModel = TopicModel(T=n_topics)
    myModel.get_text_info(processed_texts, tokens)
    C_wt, C_dt, topic_asgnmnts = myModel.train(processed_texts, n_gibbs=n_gibbs)

    # print results
    myModel.what_did_you_learn()
