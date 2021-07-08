import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
import preprocessing
import json


class ShakespearWrapper:
    def __init__(self):
        with open('token_config.json', 'r') as token_configuration:
            self.tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(
                token_configuration)

    def generate_text(self, n_chars, text):
        model = keras.models.load_model(
            './shakespear.hdf5', custom_objects={'lastOnlyMetric': preprocessing.lastOnlyMetric})
        response = preprocessing.generate_text(
            model, text, self.tokenizer, n_chars)
        return response


class DiscordModelWrapper:
    def __init__(self, temperature=1):
        with open('discord_config.json', 'r') as token_configuration:
            json_string = token_configuration.read()

        self.tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(
            json_string)
        self.tokenizer.filters = ''
        self.tokenizer.analyzer = preprocessing.p_analyzer
        self.temperature = temperature
        self.model = keras.models.load_model('disc_model.hdf5')

    def generate_response(self, input_text, max_chars=100):
        input_length = len(input_text)
        response = input_text
        response += '<START>'
        for _ in range(max_chars):
            character = self._generate_character(response)
            response += character
            if character == '<END>':
                return response[input_length:]
        response += '<END>'
        return response[input_length:]

    def _generate_character(self, text):
        X = self._preprocess([text])
        y_proba = self.model.predict(X)[0, -1:, :]
        rescaled_logits = tf.math.log(y_proba)/self.temperature
        char_id = tf.random.categorical(rescaled_logits, num_samples=1)+1
        return self.tokenizer.sequences_to_texts(char_id.numpy())[0]

    def _preprocess(self, text):
        X = np.array(self.tokenizer.texts_to_sequences(text))-1
        return X
