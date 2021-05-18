# NLP Project COMP0087 UCL

To run our inference, please visit [our Google Colab Notebook](https://colab.research.google.com/github/adamnpeace/nlp-project/blob/main/final_inference.ipynb)

To replicate our results, the following order should be used:

1. *(Optional)* Run `wiki_download.ipynb` (Note this will take a very long time (multiple days))
2. Run `data-preprocessing.ipynb` (This will run without step 1.)
3. *(Optional)* Run `train-BERT.ipynb`
4. *(Optional)* Run `train-T5.ipynb` (This will need a second virtualenv since it uses a different version of 🤗Transformers)
5. Run `eval-BERT.ipynb` (This will run without step 3.)
5. Run `eval-T5.ipynb` (This will run without step 4.)
6. Run `metrics.ipynb`

If you just want to test our models, please refer to the inference [notebook on Colab](https://colab.research.google.com/github/adamnpeace/nlp-project/blob/main/final_inference.ipynb).

*Authored by Shirin Harandi, Ksenia Pavlina, Adam Peace, Ralf Michael Yap*
