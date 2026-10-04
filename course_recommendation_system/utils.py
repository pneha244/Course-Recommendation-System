import numpy as np

def top_tfidf_terms_for_doc(tfidf_vector, feature_names, topn=5):
    if hasattr(tfidf_vector, 'toarray'):
        vec = tfidf_vector.toarray().ravel()
    else:
        vec = np.array(tfidf_vector).ravel()
    topn = min(topn, vec.shape[0])
    idxs = np.argsort(vec)[-topn:][::-1]
    terms = [(feature_names[i], float(vec[i])) for i in idxs if vec[i] > 0]
    return terms
