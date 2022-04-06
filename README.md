# WRITTEN2LATEX
# Written2LaTeX
A machine learning project aiming to turn handwritten equations into LaTeX using the [Handwritten Mathematical Expressions](https://www.kaggle.com/rtatman/handwritten-mathematical-expressions) dataset.

## Repository set-up:

To set up the repository:
- Run `pip install -U -r requirements.txt`

- If on Windows and perl is not installed, then install perl: https://www.perl.org/


### IM2LATEX-100K-HANDWRITTEN Dataset

To download the dataset:
 - Navigate to http://lstm.seas.harvard.edu/LaTeX/data/
 - Click the link labelled `IM2LATEX-100K-HANDWRITTEN.tgz (processed images, unprocessed formulas, training, validation and test set)`
 - Download to unzip such that all files/folders are directly in `/data`
- There should be 5 files and 1 folder directly in `/data`:
    -  `images`, `formulas.lst`, `test.lst`, `train.lst`, `val.lst`

To preprocess the data:
 - `cd` to the root directory `Written2LaTeX`
 - ~~Preprocess the images~~:
    - ~~`python src/im2markup/scripts/preprocessing/preprocess_images.py --input-dir data/images --output-dir data/images_processed`~~
    - Images have already been preprocessed
 - Preprocess the formulas:
    - `python scripts/preprocessing/preprocess_formulas.py --mode normalize --input-file data/formulas.lst --output-file data/formulas.norm.lst`
 - Prepare train, validation and test files:
    - `python scripts/preprocessing/preprocess_filter.py --filter --image-dir data/images --label-path data/formulas.norm.lst --data-path data/train.lst --output-path data/train_filter.lst`

    - `python scripts/preprocessing/preprocess_filter.py --filter --image-dir data/images --label-path data/formulas.norm.lst --data-path data/validate.lst --output-path data/validate_filter.lst`

    - `python scripts/preprocessing/preprocess_filter.py --no-filter --image-dir data/images --label-path data/formulas.norm.lst --data-path data/test.lst --output-path data/test_filter.lst`


## Credits
Credit to repository [im2markup](https://github.com/harvardnlp/im2markup) for the source code in `scripts`