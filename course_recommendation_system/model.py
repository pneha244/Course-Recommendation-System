import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from .data_loader import load_course_datasets
from .utils import top_tfidf_terms_for_doc

class CourseRecommender:
    def __init__(self, df=None):
        if df is None:
            df = load_course_datasets()
        self.df = df.reset_index(drop=True)
        self._fit()

    def _fit(self):
        corpus = self.df['content'].fillna('').tolist()
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=20000)
        if len(corpus) == 0:
            self.tfidf_matrix = None
            self.nn = None
            return
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
        self.nn = NearestNeighbors(n_neighbors=min(50, self.tfidf_matrix.shape[0]), metric='cosine')
        self.nn.fit(self.tfidf_matrix)

    def recommend(self, skills='', interests=None, goals='', domain=None, topk=10, filters=None):
        if interests is None:
            interests = []
        if self.tfidf_matrix is None:
            return pd.DataFrame()
        user_text = ' '.join([skills or '', ' '.join(interests), goals or '', domain or ''])
        user_vec = self.vectorizer.transform([user_text])
        cos_sim = cosine_similarity(user_vec, self.tfidf_matrix).flatten()

        # KNN-based similarity (1 - distance)
        try:
            dists, idxs = self.nn.kneighbors(user_vec, n_neighbors=min(50, self.tfidf_matrix.shape[0]))
            knn_scores = 1 - dists.flatten()
        except Exception:
            knn_scores = np.zeros_like(cos_sim)

        # Build base results
        results = self.df.copy()
        results['cosine_score'] = cos_sim
        # Map knn scores to rows (approximate by indices returned)
        knn_idx_scores = np.zeros(len(results))
        if len(idxs.shape) > 1:
            for pos, idx in enumerate(idxs.flatten()):
                # approximate assignment: add score to that index
                if pos < len(knn_scores):
                    knn_idx_scores[idx] = max(knn_idx_scores[idx], knn_scores[pos])
        results['knn_score'] = knn_idx_scores
        # hybrid score: weighted average
        results['score'] = (results['cosine_score'] * 0.6) + (results['knn_score'] * 0.4)

        # Apply filters (lenient - only if criteria met)
        if filters:
            if 'domain' in filters and filters['domain']:
                domain_filter = filters['domain'].lower()
                domain_mask = results['domain'].fillna('').str.lower() == domain_filter
                if domain_mask.sum() > 0:  # Only apply if there are matching results
                    results = results[domain_mask]
            
            if 'difficulty' in filters and filters['difficulty']:
                difficulty_filter = filters['difficulty'].lower()
                diff_mask = results['difficulty'].fillna('').str.lower() == difficulty_filter
                if diff_mask.sum() > 0:  # Only apply if there are matching results
                    results = results[diff_mask]
            
            if 'min_rating' in filters and filters['min_rating'] > 0:
                try:
                    rating_numeric = pd.to_numeric(results['rating'], errors='coerce').fillna(0)
                    rating_mask = rating_numeric >= float(filters['min_rating'])
                    if rating_mask.sum() > 0:  # Only apply if there are matching results
                        results = results[rating_mask]
                except Exception:
                    pass

        results = results.sort_values('score', ascending=False).reset_index(drop=True)

        # Build explanations
        feature_names = self.vectorizer.get_feature_names_out()
        explanations = []
        for i, row in results.iterrows():
            doc_vec = self.tfidf_matrix[i]
            terms = top_tfidf_terms_for_doc(doc_vec, feature_names, topn=5)
            explanations.append({'top_terms': terms, 'reason': f"Matches on: {', '.join([t for t, s in terms])}"})
        results['explanation'] = explanations

        return results.head(topk)
