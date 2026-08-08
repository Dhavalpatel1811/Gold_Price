# Gold Price Prediction & Investment Advisor

An academic ML project (year-end capstone) that forecasts gold price movement
from historical market data and financial news sentiment, and exposes the
results through a RAG-powered chatbot that can also answer investment-theory
questions from *The Intelligent Investor*.

## What it does

The project has three parts that build on each other:

1. **Time-series forecasting (LSTM)** — predicts the next-day gold closing
   price from the last 30 days of OHLCV data.
2. **Sentiment-based direction classifier (XGBoost + FinBERT)** — pulls the
   latest gold-market news headline, scores its sentiment with FinBERT, and
   classifies whether the price is likely to move up or down. SHAP is used
   to explain each prediction.
3. **RAG investment chatbot** — a Gradio app backed by a local LLaMA3 model
   (via Ollama) that answers three kinds of questions: general chat,
   prediction lookups (grounded in the latest output from parts 1–2), and
   investing-theory questions answered from *The Intelligent Investor*,
   retrieved with a FAISS vector index.

## Project structure

```
Gold_Price/
├── data/
│   ├── raw/                 # Source datasets (gold prices, gold news)
│   ├── processed/           # Engineered/scaled features used for training
│   └── predictions/         # Output of models/predict.py
├── models/
│   ├── train_lstm.py        # Trains the LSTM forecaster
│   ├── train_xgboost.py     # Trains the sentiment-direction classifier
│   ├── predict.py           # Runs the full hybrid prediction pipeline
│   └── artifacts/           # Saved trained models (.keras, .pkl)
├── rag_chatbot/
│   ├── rag_engine.py        # Chunking, embedding, FAISS retrieval
│   ├── chatbot.py           # Intent routing + LLaMA3 responses
│   ├── app.py                # Gradio UI — run this to launch the chatbot
│   ├── knowledge_base/      # Source book (PDF + extracted text)
│   └── embeddings/          # Cached FAISS index + chunks
├── notebooks/
│   └── news_sentiment_analysis.ipynb   # Exploratory news sentiment analysis
├── archive/                 # Earlier prototype, kept for reference
├── requirements.txt
└── .env.example
```

## Run it on your local machine

### Prerequisites
- Python 3.11+ (project was tested on Windows with a `venv`)
- [Ollama](https://ollama.com) installed, for the chatbot only
- A free API key from [newsapi.org](https://newsapi.org), for the predictor only

### 1. Clone and set up a virtual environment
```bash
git clone https://github.com/Dhavalpatel1811/Gold_Price.git
cd Gold_Price

python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
```

### 2. Add your API key
```bash
cp .env.example .env
```
Open `.env` and paste your NewsAPI key in place of `your_newsapi_key_here`.

### 3. Run the price predictor
```bash
python models/predict.py
```
This fetches the latest gold news, scores its sentiment, and produces a
combined LSTM + XGBoost prediction in `data/predictions/`.

> **If you see a Keras/LSTM loading error** ("Unrecognized keyword arguments
> passed to LSTM..."), the saved model was trained on an older TensorFlow
> version than the one installed. Retrain it locally — takes a few minutes:
> ```bash
> python models/train_lstm.py
> ```

### 4. Run the chatbot (optional)
```bash
ollama pull llama3
cd rag_chatbot
python app.py
```
Then open **http://localhost:7860** in your browser.

> **Gradio version note:** this was built against an older Gradio API.
> If `app.py` throws a `TypeError: unexpected keyword argument` on the
> `gr.Chatbot(...)` or `gr.Blocks(...)` lines, check your installed version
> with `pip show gradio` and drop the unsupported keyword arguments — Gradio
> 6.x removed `type="messages"` and `show_copy_button` from `Chatbot()`, and
> moved `css`/`theme` from `Blocks()` to `.launch()`.


## Tech stack

- **Modeling:** TensorFlow/Keras (LSTM), XGBoost, scikit-learn, SHAP
- **NLP:** FinBERT (news sentiment), sentence-transformers (embeddings)
- **RAG:** FAISS vector search over *The Intelligent Investor*
- **Chat/UI:** Ollama (LLaMA3), Gradio
- **Data:** pandas, NumPy

## Notes

- `archive/` holds an earlier single-feature LSTM prototype (using
  `yfinance` directly) from before the project moved to the curated
  feature set in `data/processed/`. Kept for reference, not maintained.
- This project is for educational purposes only and is not financial
  advice.

## Author

Dhaval Patel — [GitHub](https://github.com/Dhavalpatel1811)

## License

MIT — see [LICENSE](LICENSE).