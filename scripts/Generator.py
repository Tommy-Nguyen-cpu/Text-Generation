import tensorflow as tf

class Generator(tf.keras.Model):
    def __init__(self, model, chars_from_ids, ids_from_chars, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        self.model = model
        self.chars_from_ids = chars_from_ids
        self.ids_from_chars = ids_from_chars

    def get_texts(self, results):
        return tf.strings.reduce_join(results) # Characters are concatenated via their rows.

    @tf.function
    def generate_one_step(self, inputs, states=None):
        # Convert strings to token IDs.
        input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
        input_ids = self.ids_from_chars(input_chars).to_tensor() # Converts tokens to numeric IDs.

        # Run the model.
        # predicted_logits.shape is [batch, char, next_char_logits]
        predicted_logits, states = self.model(inputs=input_ids, states=states,
                                                return_states=True)
        # Only use the last prediction.
        predicted_logits = predicted_logits[:, -1, :]
        predicted_logits = predicted_logits/self.temperature # Determines how random or deterministic the predictions are.

        # Sample the output logits to generate token IDs.
        predicted_ids = tf.random.categorical(predicted_logits, num_samples=1) # Samples from logit distribution predicted by model.
        predicted_ids = tf.squeeze(predicted_ids, axis=-1) 

        # Convert from token ids to characters
        predicted_chars = self.chars_from_ids(predicted_ids) # Removes dimensions of size 1 (i.e. remove unnecessary dimension when it only has 1 element).

        # Return the characters and model state.
        return predicted_chars, states