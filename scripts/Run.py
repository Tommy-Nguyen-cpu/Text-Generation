import tensorflow as tf
import argparse

from Text_Gen_Model import TextGenModel

def main(args):
    text_to_file = tf.keras.utils.get_file(args.path_to_text.rsplit('/', 1)[-1], args.path_to_text)
    text = open(text_to_file, "rb").read().decode("utf-8")

    model = TextGenModel(text)

    dataset = model.create_dataset(text)
    # training_set, test_set = tf.keras.utils.split_dataset(dataset, left_size=.8)
    # print("Shape:", training_set.cardinality(), test_set.cardinality())
    model.compile(optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(args.model_save_path, save_best_only=True)
    model.fit(dataset, epochs=args.epochs, callbacks=[model_checkpoint]) # TODO: Check out why the loss is not working.

    model.save(args.model_save_path) # TODO: Figure out why, when I load the model back up, the predictions (text generation) is rubbish (random).

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Text Generation! Try it! :)")

    parser.add_argument("--path_to_text",default="https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt", type=str, help="The path to the text file we will train our model on.")
    parser.add_argument("--model_save_path", default="./", type=str, help="Path to where we will save our model.")
    parser.add_argument("--epochs", default=10, type=int, help="How many iterations our model should train for.")
    main(parser.parse_args())