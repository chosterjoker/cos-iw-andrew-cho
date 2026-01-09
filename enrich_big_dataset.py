"""
Enrich the ml-32m dataset with additional movie features from TMDb API
Including: plot synopsis, tagline, budget, revenue, runtime, cast, director, etc.

This script includes checkpointing to handle the large dataset size (~87k movies)

This script was supported by AI code generation. 

You must create a TMDB API key and add it to the script. You can get a key from https://www.themoviedb.org/settings/api.
Create a folder called data in the same directory as this script and add the ml-32m dataset to it.
"""

import pandas as pd
import requests
import time
from pathlib import Path
from tqdm import tqdm
import json

# Configuration
DATA_DIR = Path("ml-32m") # folder containing the ml-32m dataset
TMDB_API_KEY = "TMDB_API_KEY" #TMDB API key
TMDB_BASE_URL = "https://api.themoviedb.org/3"
RATE_LIMIT_DELAY = 0.021  # 48 requests/second (max safe rate)
OUTPUT_FILE = DATA_DIR / "movies_enriched_big.csv"
CHECKPOINT_FILE = DATA_DIR / "enrichment_checkpoint.json"
CHECKPOINT_INTERVAL = 100  # Save progress every 100 movies

def get_movie_full_details(tmdb_id, api_key):
    """
    Fetch comprehensive movie details from TMDb API

    Returns dict with all available movie information
    """
    if pd.isna(tmdb_id):
        return None

    url = f"{TMDB_BASE_URL}/movie/{int(tmdb_id)}"
    params = {
        'api_key': api_key,
        'append_to_response': 'credits,keywords'  # Get cast/crew and keywords in same call
    }

    try:
        response = requests.get(url, params=params, timeout=10)

        if response.status_code == 200:
            data = response.json()

            # Extract cast (top 5 actors)
            cast = data.get('credits', {}).get('cast', [])
            top_cast = [actor['name'] for actor in cast[:5]]

            # Extract director
            crew = data.get('credits', {}).get('crew', [])
            directors = [person['name'] for person in crew if person['job'] == 'Director']

            # Extract keywords
            keywords = data.get('keywords', {}).get('keywords', [])
            keyword_list = [kw['name'] for kw in keywords]

            return {
                # Plot information
                'synopsis': data.get('overview', ''),
                'tagline': data.get('tagline', ''),

                # Basic info
                'release_date': data.get('release_date', ''),
                'runtime': data.get('runtime', None),
                'original_language': data.get('original_language', ''),
                'original_title': data.get('original_title', ''),
                'status': data.get('status', ''),  # Released, Post Production, etc.

                # Financial
                'budget': data.get('budget', None),
                'revenue': data.get('revenue', None),

                # Ratings and popularity
                'vote_average': data.get('vote_average', None),
                'vote_count': data.get('vote_count', None),
                'popularity': data.get('popularity', None),

                # People
                'cast': '|'.join(top_cast) if top_cast else '',
                'director': '|'.join(directors) if directors else '',

                # Keywords/tags from TMDb
                'keywords': '|'.join(keyword_list) if keyword_list else '',

                # Production
                'production_companies': '|'.join([c['name'] for c in data.get('production_companies', [])]),
                'production_countries': '|'.join([c['iso_3166_1'] for c in data.get('production_countries', [])]),

                # Additional
                'adult': data.get('adult', False),
                'homepage': data.get('homepage', ''),
            }

        elif response.status_code == 404:
            # Movie not found - return empty dict
            return {key: '' if key in ['synopsis', 'tagline', 'release_date', 'original_language',
                                       'original_title', 'status', 'cast', 'director', 'keywords',
                                       'production_companies', 'production_countries', 'homepage']
                    else None if key in ['runtime', 'budget', 'revenue', 'vote_average',
                                         'vote_count', 'popularity']
                    else False
                    for key in ['synopsis', 'tagline', 'release_date', 'runtime', 'original_language',
                               'original_title', 'status', 'budget', 'revenue', 'vote_average',
                               'vote_count', 'popularity', 'cast', 'director', 'keywords',
                               'production_companies', 'production_countries', 'adult', 'homepage']}

        elif response.status_code == 429:
            print(f"\nRate limit hit, waiting 10 seconds...")
            time.sleep(10)
            return get_movie_full_details(tmdb_id, api_key)
        else:
            print(f"\nError {response.status_code} for tmdb_id {tmdb_id}")
            return None

    except Exception as e:
        print(f"\nException for tmdb_id {tmdb_id}: {str(e)}")
        return None

def save_checkpoint(processed_indices):
    """Save checkpoint of processed movie indices"""
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump({'processed_indices': list(processed_indices)}, f)

def load_checkpoint():
    """Load checkpoint if it exists"""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, 'r') as f:
            data = json.load(f)
            return set(data.get('processed_indices', []))
    return set()

def enrich_big_dataset():
    """Main function to enrich the ml-32m MovieLens dataset"""

    # Load data
    print("Loading ml-32m dataset...")
    movies_df = pd.read_csv(DATA_DIR / "movies.csv")
    links_df = pd.read_csv(DATA_DIR / "links.csv")

    # Merge
    movies_full = movies_df.merge(links_df, on='movieId', how='left')
    print(f"Total movies: {len(movies_full):,}")
    print(f"Movies with TMDb ID: {movies_full['tmdbId'].notna().sum():,}")

    # Initialize new columns
    new_columns = [
        'synopsis', 'tagline', 'release_date', 'runtime', 'original_language',
        'original_title', 'status', 'budget', 'revenue', 'vote_average',
        'vote_count', 'popularity', 'cast', 'director', 'keywords',
        'production_companies', 'production_countries', 'adult', 'homepage'
    ]

    for col in new_columns:
        movies_full[col] = None

    # Check for checkpoint
    processed_indices = load_checkpoint()
    if processed_indices:
        print(f"\nResuming from checkpoint: {len(processed_indices):,} movies already processed")

    # Fetch details
    print(f"\nFetching movie details from TMDb API...")
    print(f"Rate limit: ~48 requests/second")
    remaining = len(movies_full) - len(processed_indices)
    print(f"Estimated time remaining: ~{remaining * 0.021 / 60:.1f} minutes\n")

    movies_processed = 0
    for idx, row in tqdm(movies_full.iterrows(), total=len(movies_full), desc="Enriching movies"):
        # Skip if already processed
        if idx in processed_indices:
            continue

        if pd.notna(row['tmdbId']):
            details = get_movie_full_details(row['tmdbId'], TMDB_API_KEY)

            if details:
                for key, value in details.items():
                    movies_full.at[idx, key] = value

            time.sleep(RATE_LIMIT_DELAY)

        processed_indices.add(idx)
        movies_processed += 1

        # Save checkpoint periodically
        if movies_processed % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(processed_indices)
            # Also save partial results
            movies_full.to_csv(OUTPUT_FILE, index=False)

    # Save final results
    print("\n\nSaving final enriched dataset...")
    movies_full.to_csv(OUTPUT_FILE, index=False)

    # Clean up checkpoint file
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()

    # Statistics
    print(f"\n{'='*60}")
    print("DATASET ENRICHMENT COMPLETE!")
    print(f"{'='*60}")
    print(f"Total movies: {len(movies_full):,}")
    print(f"\nFeature coverage:")
    print(f"  • Synopsis: {movies_full['synopsis'].notna().sum():,} ({movies_full['synopsis'].notna().sum()/len(movies_full)*100:.1f}%)")
    print(f"  • Cast: {(movies_full['cast'].str.len() > 0).sum():,} ({(movies_full['cast'].str.len() > 0).sum()/len(movies_full)*100:.1f}%)")
    print(f"  • Director: {(movies_full['director'].str.len() > 0).sum():,} ({(movies_full['director'].str.len() > 0).sum()/len(movies_full)*100:.1f}%)")
    print(f"  • Budget: {movies_full['budget'].notna().sum():,} ({movies_full['budget'].notna().sum()/len(movies_full)*100:.1f}%)")
    print(f"  • Revenue: {movies_full['revenue'].notna().sum():,} ({movies_full['revenue'].notna().sum()/len(movies_full)*100:.1f}%)")
    print(f"  • Runtime: {movies_full['runtime'].notna().sum():,} ({movies_full['runtime'].notna().sum()/len(movies_full)*100:.1f}%)")
    print(f"  • Keywords: {(movies_full['keywords'].str.len() > 0).sum():,} ({(movies_full['keywords'].str.len() > 0).sum()/len(movies_full)*100:.1f}%)")

    print(f"\nOutput saved to: {OUTPUT_FILE}")

    # Show examples
    print(f"\n{'='*60}")
    print("SAMPLE ENRICHED MOVIES:")
    print(f"{'='*60}")

    sample = movies_full[movies_full['synopsis'].str.len() > 0].head(3)
    for idx, row in sample.iterrows():
        print(f"\n{row['title']}")
        print(f"  Synopsis: {row['synopsis'][:150]}...")
        if row['tagline']:
            print(f"  Tagline: {row['tagline']}")
        print(f"  Director: {row['director']}")
        print(f"  Cast: {row['cast']}")
        print(f"  Runtime: {row['runtime']} min")
        print(f"  Budget: ${row['budget']:,.0f}" if pd.notna(row['budget']) and row['budget'] > 0 else "  Budget: N/A")
        print(f"  Revenue: ${row['revenue']:,.0f}" if pd.notna(row['revenue']) and row['revenue'] > 0 else "  Revenue: N/A")
        print(f"  Rating: {row['vote_average']}/10 ({row['vote_count']} votes)")
        print(f"  Keywords: {row['keywords'][:100]}..." if len(str(row['keywords'])) > 100 else f"  Keywords: {row['keywords']}")

if __name__ == "__main__":
    print("TMDb Dataset Enrichment for ml-32m")
    print("=" * 60)
    print("\nThis will fetch comprehensive movie data including:")
    print("  • Plot synopsis and tagline")
    print("  • Cast (top 5 actors) and director")
    print("  • Budget and revenue")
    print("  • Runtime and release date")
    print("  • TMDb ratings and popularity")
    print("  • Keywords and production info")
    print("\nCheckpointing enabled: Progress saved every 100 movies")
    print("You can safely interrupt and resume this script\n")

    enrich_big_dataset()
