# NTSC_craw_practice

This repository contains a simple practice project for web crawling job postings from the National Science and Technology Council (NSTC) of Taiwan.

## Contents

- `NTSC_job_data.ipynb` – Jupyter notebook that fetches job posting pages and parses them with `requests` and `BeautifulSoup`.
- `nstc_jobs_partial.csv` – A partial dataset of jobs already collected as a reference.
- `RAG_mini.ipynb` – Demonstrates a lightweight retrieval-augmented generation workflow using FAISS.

## Requirements

The notebook relies on the following Python packages:

- `requests`
- `pandas`
- `beautifulsoup4`

Install them with `pip install requests pandas beautifulsoup4` if they are not already available.

### RAG_mini.ipynb Requirements

The `RAG_mini.ipynb` notebook demonstrates a minimal retrieval-augmented generation example. In addition to `pandas`, it relies on a few extra libraries:

- `langchain`
- `sentence-transformers`
- `faiss-cpu`
- `numpy`

Install them with `pip install langchain sentence-transformers faiss-cpu numpy`.

## Usage

1. Clone this repository and open `NTSC_job_data.ipynb` with Jupyter Notebook or JupyterLab.
2. Run through the cells to crawl the latest job postings from the NSTC website.
3. The notebook saves the full dataset to `nstc_jobs_full.csv` once the crawl completes.

The provided `nstc_jobs_partial.csv` file is a small snapshot of the data which can be used to test the parsing logic before performing a full crawl.

## License

This project is intended for learning purposes only. Please respect the terms of the NSTC website when running the crawler.
