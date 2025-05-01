import os
import requests
from bs4 import BeautifulSoup
import tldextract
from urllib.parse import urljoin, urlparse
from langchain.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, WebBaseLoader


DOCS_DIR = "docs"

pdf_paths = [
    os.path.join(DOCS_DIR, "America's_Choice_2500_Gold_SOB.pdf"),
    os.path.join(DOCS_DIR, "America's_Choice_5000_Bronze_SOB.pdf"),
    os.path.join(DOCS_DIR, "America's_Choice_5000_HSA_SOB.pdf"),
    os.path.join(DOCS_DIR, "America's_Choice_7350_Copper_SOB.pdf")
]

pdf_documents = []
for path in pdf_paths:
    print(f"Loading PDF: {path}")
    loader = PyPDFLoader(path)
    pdf_documents.extend(loader.load())

docx_path = os.path.join(DOCS_DIR, "America's_Choice_Medical_Questions_-_Modified.docx")
print(f"Loading DOCX: {docx_path}")
docx_loader = UnstructuredWordDocumentLoader(docx_path)
docx_documents = docx_loader.load()

all_documents = pdf_documents + docx_documents
print(f"Loaded {len(all_documents)} documents from local files.")

visited = set()
to_visit = ["https://www.angelone.in/support"]
domain = "angelone.in"

def is_valid_url(url):
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and tldextract.extract(url).domain == tldextract.extract(domain).domain

def crawl_site(start_urls, max_pages=100):
    discovered_urls = []
    while to_visit and len(discovered_urls) < max_pages:
        current_url = to_visit.pop()
        if current_url in visited:
            continue
        visited.add(current_url)

        try:
            response = requests.get(current_url, timeout=10)
            if response.status_code != 200:
                continue
            soup = BeautifulSoup(response.text, 'html.parser')
            discovered_urls.append(current_url)

            for link in soup.find_all('a', href=True):
                abs_url = urljoin(current_url, link['href'])
                if abs_url not in visited and abs_url.startswith("https://www.angelone.in/support"):
                    to_visit.append(abs_url)

        except requests.RequestException:
            continue
    return discovered_urls

print("Crawling Angel One support section...")
support_urls = crawl_site(to_visit, max_pages=100)
print(f"Discovered {len(support_urls)} support URLs.")

web_loader = WebBaseLoader(support_urls)
web_documents = web_loader.load()
print(f"Loaded {len(web_documents)} documents from webpages.")

all_documents += web_documents
print(f"Total combined documents: {len(all_documents)}")

print("\nSample document preview:")
print(all_documents[0].page_content[:500])


