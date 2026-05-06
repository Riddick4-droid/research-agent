from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

def combine_papers(papers):
    """Combine multiple papers into a single text."""
    combined_text = "\n\n".join([f"Title: {paper['title']}\nAuthors: {', '.join(paper['authors'])}\nPublished: {paper['published']}\nSummary: {paper['summary']}\nURL: {paper['url']}" for paper in papers])
    return combined_text

def split_text(text):
    """Split text into chunks."""
    return splitter.split_text(text)

def remove_duplicates(texts):
    """Remove duplicate texts."""
    seen = set()
    unique_texts = []
    for text in texts:
        if text not in seen:
            unique_texts.append(text)
            seen.add(text)
    return unique_texts