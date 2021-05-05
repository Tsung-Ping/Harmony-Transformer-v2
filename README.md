# Harmony-Transformer-v2

An improved version of the [Harmony Transformer](https://archives.ismir.net/ismir2019/paper/000030.pdf). We evaluated the new model in terms of automatic chord recognition for symbolic music. For more details, please refer to ["Attend to Chords: Improving Harmonic Analysis of Symbolic Music Using Transformer-Based Models" (TISMIR 2021)](https://transactions.ismir.net/articles/10.5334/tismir.65/).


## File descriptions
 * `BPS_FH_preprocessing.py`: preprocessing of the BPS-FH dataset

 * `chord_recognition_models.py`: implementations of the three models in comparison: the Harmony Transformer (HT/HTv2), the Bi-directional Transformer for Chord Recognition (BTC), and a convolutional recurrent neural network (CRNN)

 * `chord_symbol_recognition.py`: chord recognition using 24 maj-min chord vocabulary 

 * `functional_harmony_recognition.py`:  chord recognition using vocabulary of Roman numeral (RN) analysis

