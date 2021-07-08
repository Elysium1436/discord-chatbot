import tensorflow as tf
from tensorflow import keras
import numpy as np
import re
from functools import partial


def lastOnlyMetric(y_true, y_pred):
    return keras.metrics.categorical_accuracy(y_true[:, -1], y_pred[:, -1])


def preprocess(text, tokenizer):
    X = np.array(tokenizer.texts_to_sequences(text))-1
    max_id = len(tokenizer.word_index)
    X = tf.one_hot(X, max_id)
    return X


def analyzer(text, tokens=[], lower=False, character_white_list_regex=r'\w.,:!? '):
    """
    Creates a text analyzer to distinguish between characters and special-tokens.
    """
    tokens_list_regex = [r'(?:{})'.format(token) for token in tokens]
    tokens_re = '|'.join(tokens_list_regex)
    text = re.sub(r'[ ]+', ' ', text)
    l = re.findall(r'({}|(?:[{}]))'.format(
        tokens_re, character_white_list_regex), text)
    l = np.array(l)
    if lower:
        mask = ~np.isin(l, tokens)
        l[mask] = np.char.lower(l[mask])
    return l


p_analyzer = partial(analyzer, tokens=[
                     '<ENTER>', '<END>', '<START>'], lower=True, character_white_list_regex=r'\w.,:!? ')


def generate_character(model, text, tokenizer, temperature=1):
    X_new = preprocess([text], tokenizer)
    y_proba = model.predict(X_new)[0, -1:, :]
    rescaled_logits = tf.math.log(y_proba)/temperature
    char_id = tf.random.categorical(rescaled_logits, num_samples=1)+1
    return tokenizer.sequences_to_texts(char_id.numpy())[0]


def generate_text(model, text, tokenizer, n_chars=50, temperature=0.45):
    new_text = text
    for _ in range(n_chars):
        new_text += generate_character(model, new_text,
                                       tokenizer, temperature=temperature)
    return new_text
