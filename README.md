# RPG Maker Decrypter

## What is it?

This is a Python script which can be used to extract files from an RPG Maker XP encrypted archive. These archives have the extension `.rgssad` and are created when you choose to encrypt the game data when saving a project in RPG Maker XP.

However, there is an existing program which does the same thing available at [https://github.com/uuksu/RPGMakerDecrypter](https://github.com/uuksu/RPGMakerDecrypter). That program is pretty much better than this one in every way as it's been developed for longer, supports more variants on the format, has a user-friendly GUI, is a lot faster, etc. This program exists only because I wanted to write my own version as an exercise. You probably shouldn't actually use it.

## Installation

Clone the repository, set your current working directory to the root directory of the repository, and run the following commands, or their equivalents:

    py -m venv env
    env\Scripts\pip install -r requirements.txt

## Running the program

    env\Scripts\python -m rpgmaker_decrypter <input_file> <output_dir>

Options:

* `-v`, `--verbose`: prints messages to standard output showing you the names of the files in the archive that are being decrypted as they are being decrypted.
* `-o`, `--overwrite`: makes so that if any of the file names specified in the archive already exists in the output directory, the file will be overwritten automatically, without the program asking the user for confirmation.
* `-p`, `--profile`: makes it so that only the first 100 files in the archive will be decrypted. Useful to make the running time of the program shorter so you can profile it or test it more quickly.

## Testing

To test the program, take a `.rgssad` file and use uuksu's [RPGMakerDecrypter](https://github.com/uuksu/RPGMakerDecrypter) tool to decrypt it, extracting the files to the directory `example/expected-output`. Then run pytest:

    env\Scripts\pytest

This will compare the results from this program with those stored in the `example/expected-output` directory.

## Profiling

To profile the program:

    ./profile

This uses cProfile. Results will be written to `profile-op.txt`.