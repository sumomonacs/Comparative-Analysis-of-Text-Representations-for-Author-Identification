'''
    This file downloads the books and preprocess them for training and testing dataset.
'''
import os
import re
import sys
import requests
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.config import RAW_DIR, CLEANED_DIR, EVALUATE_RAW_DIR, EVALUATE_CLEANED_DIR

# get the raw text files from project Gutenberg
books = {
    # Charlotte Brontë
    "bronte_jane_eyre.txt": "https://www.gutenberg.org/files/1260/1260-0.txt",
    "bronte_villette.txt": "https://www.gutenberg.org/files/9182/9182-0.txt",

    # George Eliot
    "eliot_mill_on_the_floss.txt": "https://www.gutenberg.org/files/6688/6688-0.txt",
    "eliot_middlemarch.txt": "https://www.gutenberg.org/files/145/145-0.txt",

    # Henry James
    "james_turn_of_the_screw.txt": "https://www.gutenberg.org/files/209/209-0.txt",
    "james_portrait_of_a_lady.txt": "https://www.gutenberg.org/files/2833/2833-0.txt",

    # Virginia Woolf
    "woolf_mrs_dalloway.txt": "https://www.gutenberg.org/files/63107/63107-0.txt",
    "woolf_to_the_lighthouse.txt": "https://www.gutenberg.org/files/144/144-0.txt",  
    "woolf_orlando.txt" : "https://gutenberg.net.au/ebooks02/0200331.txt",

    # Edith Wharton
    "wharton_age_of_innocence.txt": "https://www.gutenberg.org/files/421/421-0.txt",
    "wharton_house_of_mirth.txt": "https://www.gutenberg.org/files/284/284-0.txt"
}

# unseen test data from Project Gutenberg:
books_unseen = {
    # Charlotte Brontë
    "bronte_shirley.txt": "https://www.gutenberg.org/files/30486/30486-0.txt", 
    "bronte_the_professor.txt": "https://www.gutenberg.org/cache/epub/1028/pg1028.txt",
    # George Eliot
    "eliot_silas_marner.txt": "https://www.gutenberg.org/files/550/550-0.txt", 
    "eliot_adam_bede.txt":"https://www.gutenberg.org/cache/epub/507/pg507.txt",

    # Henry James
    "james_the_ambassadors.txt": "https://www.gutenberg.org/cache/epub/432/pg432.txt", 
    "james_the_beast_in_the_jungle.txt": "https://www.gutenberg.org/cache/epub/1093/pg1093.txt",

    # Virginia Woolf
    "woolf_night_and_day.txt": "https://www.gutenberg.org/files/1245/1245-0.txt",  
    "woolf_the_voyage_out.txt": "https://www.gutenberg.org/cache/epub/144/pg144.txt",

    # Edith Wharton
    "wharton_the_reef.txt": "https://www.gutenberg.org/cache/epub/283/pg283.txt", 
    "wharton_ethan_frome.txt": "https://www.gutenberg.org/cache/epub/4517/pg4517.txt"
}

# openings of the books
OPENINGS = {
    # Charlotte Brontë
    "bronte_jane_eyre.txt": "There was no possibility of taking a walk that day",
    "bronte_villette.txt": "My godmother lived in a handsome house in the clean and ancient town ",

    # George Eliot
    "eliot_mill_on_the_floss.txt": "A wide plain, where the broadening Floss hurries on",
    "eliot_middlemarch.txt": "Miss Brooke had that kind of beauty which seems to be thrown",

    # Henry James
    "james_turn_of_the_screw.txt": "The story had held us, round the fire, sufficiently breathless,",
    "james_portrait_of_a_lady.txt": "Under certain circumstances there are few hours in life more agreeable",

    # Virginia Woolf
    "woolf_mrs_dalloway.txt": "Mrs Dalloway said she would buy the gloves herself.",
    "woolf_to_the_lighthouse.txt": "As the streets that lead from the Strand to the Embankment are very",
    "woolf_orlando.txt" : "He--for there could be no doubt of his sex,",
    
    # Edith Wharton
    "wharton_age_of_innocence.txt": "I will begin the story of my adventures with a certain morning early",
    "wharton_house_of_mirth.txt": "Selden paused in surprise. In the afternoon rush of the Grand",
}

# openings of the unseen books
OPENINGS_UNSEEN = {
    # Charlotte Brontë
    "bronte_shirley.txt": "Of late years an abundant shower of curates has fallen upon",
    "bronte_the_professor.txt": "THE other day, in looking over my papers,",

    # George Eliot
    "eliot_silas_marner.txt": "In the days when the spinning-wheels hummed busily in the",
    "eliot_adam_bede.txt": "With a single drop of ink for a mirror,",

    # Henry James
    "james_the_ambassadors.txt": "Strether’s first question, when he reached the hotel",
    "james_the_beast_in_the_jungle.txt": "What determined the speech that startled him in the course of their",

    # Virginia Woolf
    "woolf_night_and_day.txt": "It was a Sunday evening in October,",
    "woolf_the_voyage_out.txt": "As the streets that lead from the Strand to the Embankment",
    
    # Edith Wharton
    "wharton_the_reef.txt":"Unexpected obstacle. Please don’t come till thirtieth",
    "wharton_ethan_frome.txt": "I had the story, bit by bit, from various people,"
}

# preprocess the book
def clean_book(file_path, opening_line):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    # remove Gutenberg header/footer
    start = re.search(r"\*\*\* *START OF.*?\*\*\*", text, re.IGNORECASE)
    end = re.search(r"\*\*\* *END OF.*?\*\*\*", text, re.IGNORECASE)
    if start and end:
        text = text[start.end():end.start()]
    elif start:
        text = text[start.end():]
    elif end:
        text = text[:end.start()]

    # trim before true narrative opening
    idx = text.find(opening_line)
    if idx != -1:
        text = text[idx:]
    else:
        print(f"Opening line not found in: {file_path}")

    # remove multiline [Illustration: ... ] or [ ... ] blocks
    text = re.sub(r"\[\s*Illustration:.*?\]+", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"\[\s*_?Copyright.*?\]+", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"\[\s*.*?\s*\]", "", text, flags=re.DOTALL)  # generic fallback

    return text.strip()

# dowmload the books to the directory data
def download_books(book, dir):
    for fname, url in book.items():
        out_path = os.path.join(dir, fname)
        if os.path.exists(out_path):
            print(f"Skipping {fname} (already exists)")
            continue
        print(f"Downloading {fname} ...")
        try:
            r = requests.get(url, timeout=15)
            r.raise_for_status()
        except requests.RequestException as e:
            print(f"Failed to download {fname}: {e}")
            continue
        
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(r.text)
    
    print(f"All books processed into {dir}")

# get the cleaned version of the books
def get_cleaned(input, output, openings):
    for fname, opening in openings.items():
        in_path = os.path.join(input, fname)
        out_path = os.path.join(output, fname)
        if os.path.exists(in_path):
            cleaned = clean_book(in_path, opening)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(cleaned)
            print(f"Cleaned: {fname}")
        else:
            print(f"File not found: {in_path}")

def main():
    download_books(books, RAW_DIR)
    download_books(books_unseen, EVALUATE_RAW_DIR)
    get_cleaned(RAW_DIR, CLEANED_DIR, OPENINGS)
    get_cleaned(EVALUATE_RAW_DIR, EVALUATE_CLEANED_DIR, OPENINGS_UNSEEN)

if __name__ == "__main__":
    main()