# Housing Rental Price Prediction: An AI Engineering Project

This project predicts housing rental prices across the U.S. using a robust **GenAI Data Pipeline** and supervised machine learning models. Built to operate on large-scale, unstructured real-estate advertisements, the pipeline extracts, structures, and semantically clusters data before modeling.

## 🚀 Key Features

*   **Local LLM Inference:** Deployed a 4-bit quantized `google/gemma-3-4b-it` model on resource-constrained hardware (Tesla T4) to extract complex, structured JSON payloads directly from unstructured ad copy.
*   **Prompt Engineering & Fallbacks:** Engineered few-shot prompts with strict schema enforcement to categorize amenities (e.g., floor finishes, outdoor spaces) and extract dynamic pricing/unit dimension ranges. Handled edge-cases through robust python parsing fallback structures.
*   **Semantic Search & Unsupervised NLP:** Utilized `google/embeddinggemma-300m` via `SentenceTransformers` to convert 13,000+ extracted key phrases into 768-D vectors. Applied **UMAP** dimensionality reduction and **HDBSCAN** clustering to discover 25 distinct semantic features across the housing market ad-space.
*   **Machine Learning Price Prediction:** Built ensemble predictive models (Random Forest, XGBoost) using the structured metrics extracted by the LLM, alongside geospatial intelligence (lat/long mappings). Random Forest achieved the highest predictive accuracy of 89%.
*   **Fault-Tolerant Batching:** Developed asynchronous-style batch inference loops with incremental checkpointing to gracefully handle potential OOM errors over thousands of rows.

## 📂 Repository Structure

*   **/Housing rental Prices app/Information Retrieval**: Core NLP notebooks detailing the Gemma 4-bit quantization setup, information extraction logic, and UMAP/HDBSCAN clustering logic.
*   **/Housing rental Prices app/EDA & Modelling**: Machine learning pipelines for training the Random Forest and XGBoost predictive models on the extracted data.
*   **/Housing rental Prices app/price_prediction_app**: A Streamlit/Dash interactive interface demonstrating the predictive capabilities based on user-driven amenity/location configurations.

## 🛠 Tech Stack

**AI & NLP:** Hugging Face `transformers`, `SentenceTransformers`, `bitsandbytes` (4-bit quantization), `accelerate`, UMAP, HDBSCAN  
**Machine Learning:** Scikit-Learn, XGBoost, Pandas, Numpy, SHAP (Explainable AI)  
**App Deployment:** Streamlit / Dash  

*(Note: Raw 100MB+ datasets and generated `44MB` model `.pkl` files have been omitted from this repository for lightweight tracking. The underlying data generation process is fully reproducible via the notebooks.)*
