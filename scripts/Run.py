import tensorflow as tf
import argparse

from Text_Gen_Model import TextGenModel
from Generator import Generator

def generate_text(generator, text, number_of_characters = 1000):
    next_char = tf.constant([text])
    states = None
    results = [next_char]

    for i in range(number_of_characters):
        next_char, states = generator.generate_one_step(next_char, states)
        results.append(next_char)
    return tf.strings.reduce_join(results).numpy().decode("UTF-8")

def main(args):
    if args.randomness <= 0 or args.randomness < 1:
        print("RANDOMNESS MUST BE GREATER THAN 0 and LESS THAN OR EQUAL TO 1. DEFAULTING TO .1 SINCE INPUT WAS OUTSIDE THAT RANGE.")
        args.randomness = 0.1
    if args.train:
        text_to_file = tf.keras.utils.get_file(args.path_to_text.rsplit('/', 1)[-1], args.path_to_text)
        text = open(text_to_file, "rb").read().decode("utf-8")

        model = TextGenModel(text)

        dataset = model.create_dataset(text)

        model.compile(optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(args.model_save_path, save_best_only=True)
        model.fit(dataset, epochs=args.epochs, callbacks=[model_checkpoint])

        generator = Generator(model, model.chars_from_ids, model.ids_from_chars, args.randomness)

        print("Testing Generator...\n", "--"*80, generate_text(generator,"ROMEO:")) # We need to run our generator at least once so it can be built.
        tf.saved_model.save(generator, args.model_save_path)
    else:
        model = tf.saved_model.load(args.model_load_path)
        print('Generating text...\n', "--"*80)
        print(generate_text(model, args.starter_text))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Text Generation! Try it! :)")
    parser.add_argument("--train", action=argparse.BooleanOptionalAction, help="Whether we should train a new model or not. If this flag is not called, it will load a model at the path specified by 'model_load_path'.")
    parser.add_argument("--path_to_text",default="https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt", type=str, help="The path to the text file we will train our model on.")
    parser.add_argument("--model_save_path", default="./saved_model", type=str, help="Path to where we will save our model.")
    parser.add_argument("--model_load_path", default="./saved_model", type=str, help="Path to the model to load.")
    parser.add_argument("--epochs", default=10, type=int, help="How many iterations our model should train for.")



    parser.add_argument("--randomness", default=0.1, type=float, help="Chose values between 0 and 1, determining how random the results will be (0.1 is deterministic and 1 is more random).")
    parser.add_argument("--starter_text", default="ROMEO:", type=str, help="Input to the model to generate text.")
    main(parser.parse_args())