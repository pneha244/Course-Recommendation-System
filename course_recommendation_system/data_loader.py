import os
import glob
import pandas as pd

def _normalize_columns(df):
    # map common column names to a standard set
    col_map = {}
    lower = {c.lower(): c for c in df.columns}
    def pick(*names):
        for n in names:
            if n in lower:
                return lower[n]
        return None
    title = pick('title', 'course_title', 'name')
    desc = pick('description', 'summary', 'about')
    domain = pick('domain', 'category', 'subject')
    difficulty = pick('difficulty', 'level', 'course_difficulty')
    duration = pick('duration', 'length', 'content_duration')
    rating = pick('rating', 'avg_rating', 'score', 'course_rating')
    url = pick('url', 'link')
    
    if title:
        col_map[title] = 'title'
    if desc:
        col_map[desc] = 'description'
    if domain:
        col_map[domain] = 'domain'
    if difficulty:
        col_map[difficulty] = 'difficulty'
    if duration:
        col_map[duration] = 'duration'
    if rating:
        col_map[rating] = 'rating'
    if url:
        col_map[url] = 'url'
    df = df.rename(columns=col_map)
    
    # Fill missing description with synthetic text from other fields
    if 'description' not in df.columns or df['description'].isna().all():
        parts = []
        for col in ['title', 'domain', 'difficulty']:
            if col in df.columns:
                parts.append(df[col].fillna('').astype(str))
        if parts:
            df['description'] = parts[0]
            for p in parts[1:]:
                df['description'] = df['description'] + ' ' + p
    
    return df

def load_course_datasets(base_dir=None):
    """Search for CSV files near this package and load them into a single DataFrame."""
    if base_dir is None:
        base_dir = os.path.dirname(__file__)
    # search current folder and up to two levels up
    search_paths = [base_dir, os.path.abspath(os.path.join(base_dir, '..'))]
    dfs = []
    for p in search_paths:
        pattern = os.path.join(p, '*.csv')
        for f in glob.glob(pattern):
            try:
                df = pd.read_csv(f)
                df = _normalize_columns(df)
                df['source_file'] = os.path.basename(f)
                dfs.append(df)
            except Exception:
                continue
    if not dfs:
        # fallback: try workspace root
        root = os.path.abspath(os.path.join(base_dir, '..', '..'))
        for f in glob.glob(os.path.join(root, '*.csv')):
            try:
                df = pd.read_csv(f)
                df = _normalize_columns(df)
                df['source_file'] = os.path.basename(f)
                dfs.append(df)
            except Exception:
                continue
    if not dfs:
        # return empty df with expected columns
        return pd.DataFrame(columns=['title','description','domain','difficulty','duration','rating','url','source_file'])
    out = pd.concat(dfs, ignore_index=True, sort=False)
    # ensure required columns
    for c in ['title','description','domain','difficulty','duration','rating','url']:
        if c not in out.columns:
            out[c] = None
    # create a content column
    out['content'] = (out['title'].fillna('') + ' ' + out['description'].fillna('') + ' ' + out['domain'].fillna(''))
    return out
