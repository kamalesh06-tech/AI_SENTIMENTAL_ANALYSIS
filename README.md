# AI_SENTIMENTAL_ANALYSIS
NLP-based product feedback analyzer using a pre-trained DistilBERT model. Supports URL scraping and CSV upload, sentiment classification, scoring, and decision support.


# AI Product Review Analyzer

An AI-powered web application that analyzes genuine customer reviews from product URLs or CSV files using NLP.

## Features
- Extracts real customer reviews from any product URL
- just by a product url it can analyse the reviews and result it and it also mention the product name and the market of the product(ie..amazon,flipkart,shopify)
- Not only analyse the it also analyse the emoji by the tockenization
- Filters offers, prices, delivery info, and non-review content
- Sentiment analysis using pre-trained DistilBERT
- Classifies reviews into Positive, Negative, and Neutral
- Displays results in a clean table format
- Provides a product score (out of 10) and buy/not-buy recommendation
- Built with Python and Streamlit
- Here app.py , sentiment_analysis.py,requirement.txt are core files

## Tech Stack
- Python
- Streamlit
- HuggingFace Transformers
- DistilBERT
- Pandas
- BeautifulSoup

- Ai-product-review-analyzer/
```text
ai-product-review-analyzer/
│
├── app.py                    # Main Streamlit application (UI + logic)
├── sentiment_analysis.py     # Core sentiment analysis logic
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
├── LICENSE                   # MIT License
│
├── screenshots/              # Project UI screenshots
│   ├── home.png              # Home screen
│   ├── analysis.png          # Sentiment analysis summary
│   └── reviews_table.png     # Classified reviews table
│
└── data/                     # sample csv review folder
```



## How to Run
```bash
streamlit run app.py
