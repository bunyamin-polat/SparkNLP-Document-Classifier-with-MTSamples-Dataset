# Medical Document Classification with Spark NLP (MTSamples)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bunyamin-polat/SparkNLP-Document-Classifier-with-MTSamples-Dataset/blob/main/MTSamples_Document_Classifier.ipynb)

This project classifies medical transcriptions (operative notes, consultations, imaging reports) into their **medical specialty**: Gastroenterology, Neurology, Orthopedic, Radiology or Urology. Six pipelines are built in Spark NLP and compared on the same task: classic bag-of-words features, clinical word embeddings, BioBERT and the Universal Sentence Encoder.

## Results

Evaluated on a held-out test set of 139 documents (20% of the data):

| Pipeline | Accuracy | Macro F1 |
| --- | ---: | ---: |
| **TF-IDF + Logistic Regression** | **0.81** | **0.79** |
| CountVectorizer + Logistic Regression | 0.80 | 0.78 |
| Universal Sentence Encoder + `ClassifierDL` | 0.73 | 0.73 |
| Clinical word embeddings + Logistic Regression | 0.72 | 0.70 |
| Clinical word embeddings + `ClassifierDL` | 0.60 | 0.49 |
| BioBERT (`biobert_pubmed_base_cased`) + `ClassifierDL` | 0.22 | 0.07 |

**Findings:**

- On this small dataset (818 documents), **sparse bag-of-words features with a linear model performed best**. Specialty vocabulary such as "colonoscopy" or "MRI" carries most of the signal.
- Averaging word or sentence embeddings over long documents dilutes that vocabulary. The deep-learning classifier on clinical embeddings never predicted Urology (recall 0.00).
- The BioBERT run **collapsed to a single class** (it predicted Orthopedic for every document). This is a failed training run, not a measure of what BioBERT can do; it would need tuning or fine-tuning to be a fair comparison.
- The train-set accuracy of the bag-of-words models is about 0.90, compared with about 0.81 on the test set, which shows moderate overfitting.

## Approach

**Preprocessing (Spark NLP):** `DocumentNormalizer` removes HTML tags and empty documents. Then the pipeline tokenises, normalises (removes punctuation) and removes stop words using a **clinical stop-word list** instead of a general one. Finally it lemmatises with `lemma_antbnc`.

**Pipelines compared:**

| Features | Classifier |
| --- | --- |
| `CountVectorizer` (term counts; vocabulary 10,000, `minDF=5`) | Spark ML `LogisticRegression` (`maxIter=10`, `regParam=0.3`) |
| `HashingTF` (10,000 features) + `IDF` (`minDocFreq=5`) | Spark ML `LogisticRegression` |
| `embeddings_clinical` averaged into document vectors | Spark ML `LogisticRegression` |
| `embeddings_clinical` → `SentenceEmbeddings` | Spark NLP `ClassifierDL` (200 epochs, lr 0.003) |
| `biobert_pubmed_base_cased` → `SentenceEmbeddings` | Spark NLP `ClassifierDL` (64 epochs, lr 0.003) |
| Universal Sentence Encoder | Spark NLP `ClassifierDL` (300 epochs, lr 0.003) |

Environment: Spark NLP 3.4.0 and Spark NLP for Healthcare 3.4.0 on Apache Spark 3.1.2, in Google Colab.

## Data

**MTSamples:** sample medical transcription reports published at [mtsamples.com](https://www.mtsamples.com/). The samples are already de-identified and contain no real patient data. Five specialties were selected:

| Specialty | Documents |
| --- | ---: |
| Orthopedic | 222 |
| Radiology | 188 |
| Gastroenterology | 157 |
| Neurology | 141 |
| Urology | 110 |
| **Total** | **818** |

The data is split 80/20 with a fixed seed: 679 training and 139 test documents. `mtsamples.csv` and the cleaned `mtsamples .parquet` are included in the repository. The clinical stop-word list comes from [kavgan/clinical-concepts](https://github.com/kavgan/clinical-concepts).

## How to Run

1. Open the notebook in Google Colab.
2. Upload your **Spark NLP for Healthcare license JSON** when the first cell asks for it. The clinical embeddings and `sparknlp_jsl` require a license from John Snow Labs; no license keys are stored in this repository.
3. Change the data path in the "Read MTSamples Data" cell to the location of `mtsamples .parquet`, then run all cells.

The CountVectorizer, TF-IDF, BioBERT and Universal Sentence Encoder pipelines use only the open-source Spark NLP library and can run without a license after removing the healthcare imports.

## Limitations

- **Small dataset and a single split.** With 139 test documents, differences of a few points are within noise; cross-validation would give more reliable estimates.
- **Two pipelines used a different test split.** All splits are 80/20 with the same seed, but they were applied to different DataFrames. The clinical-embeddings + `ClassifierDL` and Universal Sentence Encoder pipelines were therefore evaluated on a different set of 139 documents (class supports differ slightly), so their scores are not strictly comparable with the other rows.
- No hyperparameter search was performed. Better results are expected from tuned models or from fine-tuned transformer classifiers.
- Software versions are from early 2022 and may need small API changes on current releases.

## License

The code and notebooks are released under the [MIT License](LICENSE). The MTSamples data files are not covered by this license; they remain under the terms of their original source (see Data).
