# RPG Maker Decrypter

## What is it?

This is a Python script which can be used to extract files from an RPG Maker XP encrypted archive. These archives have the extension `.rgssad` and are created when you choose to encrypt the game data when saving a project in RPG Maker XP.

However, there is an existing program which does the same thing available at [https://github.com/uuksu/RPGMakerDecrypter](https://github.com/uuksu/RPGMakerDecrypter). That program is pretty much better than this one in every way as it's been developed for longer, supports more variants on the format, has a user-friendly GUI, is a lot faster, etc. This program exists only because I wanted to write my own version as an exercise. You probably shouldn't actually use it.

## Installation

Clone the repository, set your current working directory to the root directory of the repository, and run the following commands, or their equivalents:

    py -m venv env
    env\Scripts\pip install -r requirements.txt

Also run the following command to create the `crpgmaker_decrypter` module:

    ./cythonize

## Running the program

    env\Scripts\python -m rpgmaker_decrypter <input_file> <output_dir>

Replace `<input_file>` with the path to the `.rgssad` archive and replace `<output_dir>` with the path to the directory the decrypted files will be extracted to.

Options:

* `-o`, `--overwrite`: overwrite files in the output directory if they have the same name as one of the decrypted files (redundant if `-w` is used)
* `-w`, `--wipe`: wipe output directory completely before writing into it
* `-v`, `--verbose`: print messages to standard output showing which files are being decrypted
* `-p`, `--profile`: only decrypt a limited number of files, to make the program run quicker when profiling/testing

## Testing

To test the program, take a `.rgssad` file and use uuksu's [RPGMakerDecrypter](https://github.com/uuksu/RPGMakerDecrypter) tool to decrypt it, extracting the files to the directory `example/expected-output`. Then run pytest:

    env\Scripts\pytest

This will compare the results from this program with those stored in the `example/expected-output` directory.

## Profiling

To profile the program:

    ./profile

This uses cProfile. Results will be written to `profile-op.txt`.