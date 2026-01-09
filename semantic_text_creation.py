# this function is what turns the movie enriched data set into the structued semantic text dataset. 
# NO NEED TO RUN THIS CODE, IT IS JUST FOR REFERENCE
import re

movies_df['clean_title'] = movies_df['title'].astype(str).str.replace(r'\s*\(\d{4}\)$', '', regex=True)

def create_semantic_text_clean(row):
    # truncate synopsis to 200 words
    synopsis_val = str(row.get('synopsis', ''))
    synopsis_words = synopsis_val.split()
    if len(synopsis_words) > 200:
        synopsis_truncated = ' '.join(synopsis_words[:200])
    else:
        synopsis_truncated = synopsis_val

    genres = row.get('genres', '')
    keywords = row.get('keywords', '')
    tagline = row.get('tagline', '')

    text = f"Title: {row['clean_title']}. Genres: {genres}. Keywords: {keywords}. Tagline: {tagline}. Synopsis: {synopsis_truncated}"
    return text
