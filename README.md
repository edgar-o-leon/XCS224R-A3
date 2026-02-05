# XCS Student Code Repository
This repository contains all code for your assignment!
The build tools in this repo can be used to run the autograder locally or
compile a LaTeX submission.

## Development environment

We provide a single local development environments that can be used with Azure cloud for GPU access.

For the local environment, you need to use either `uv` or `conda` to set up your Python environment. For assignment notebooks that require **GPU**, you can use the Azure cloud VM (virtual machine) provided to you. The advantages of setting up your own environment are: (i) you can use your preferred IDE such as VSCode, and (ii) you can run basic test cases locally to help with debugging.

The followings describes how to setup the local environments:

### Local + `uv` (preferred)
For assignment development, it is best if you work on a local environment with `uv` as your Python package manager. We also have legacy support for `conda`. If you wish to rely on `conda` [please follow setup instructions here](#option-2-using-conda-legacy).
- [How to setup `uv`?](#option-1-using-uv-recommended)
- [How to run the autograder?](#running-the-autograder-locally)

## Setting Up Virtual Environment

Here is how you setup your local environment.
There are two ways to set up and manage the Python environment for this project:

### Option 1: Using uv (Recommended)
We have introduced [uv](https://docs.astral.sh/uv/) for a modern, faster environment management experience. For more detailed setup instructions, please refer to [the uv setup guide](docs/uv_setup.md).

This workflow uses:
- `pyproject.toml` to define base dependencies
- `requirements.txt` for CPU and MPS (Apple GPU) systems
- `requirements.cuda.txt` for CUDA (Nvidia GPUs)-enabled systems

#### Installation Steps
1. Run the installation script:
    ```bash
    source install.sh
    ```
    This will:
    - Create a virtual environment in the root directory named `.venv`
    - Configure OS compatible python version in `.python-version`
    - Sync dependencies from `pyproject.toml`
    - Automatically install either CPU or CUDA requirements depending on your hardware
2. Activate the environment
    ```bash
    source .venv/bin/activate
    ```

    > [!IMPORTANT]  
    > For every new terminal session you will need to activate your virtual environment. 

3. Deactivate
    ```bash
    deactivate
    ```

You can check if your virtual environment is ready for use by running `which python` and ensuring that the path returned is coming from within your `.venv/bin` directory. 

### Option 2: Using conda (Legacy)
If you prefer using [Conda](https://anaconda.org/anaconda/conda), please walk through the
[Anaconda Setup for XCS Courses](https://github.com/scpd-proed/General_Handouts/blob/master/Anaconda_Setup.pdf) to familiarize yourself with the coding environment. You can create the environment from the provided  `environment.yml` and/or `environment_cuda.yml` file located in the `/src` directory:

```bash
cd src
conda env create -f environment.yml
# if GPU/CUDA support available
conda env create -f environment_cuda.yml
conda activate <env_name>
```

Replace `<env_name>` with the name specified in the `environment.yml` file.

Deactivate the environment at any time with:
```bash
conda deactivate
```


## Running the autograder locally
All assignment code is in the `src/submission.py` file or `src/submission/` folder.  Please only make changes between the lines containing
`### START CODE HERE ###` and `### END CODE HERE ###`.

The unit tests in `src/grader.py` will be used to autograde your submission.
Run the autograder locally using the following terminal command within the
`src/` subdirectory:
```
(XCS_ENV) $ python grader.py
```

There are two types of unit tests used by our autograders:
- `basic`:  These unit tests will verify only that your code runs without
  errors on obvious test cases.

- `hidden`: These unit tests will verify that your code produces correct
  results on complex inputs and tricky corner cases.  In the student version of
  `src/grader.py`, only the setup and inputs to these unit tests are provided.
  When you run the autograder locally, these test cases will run, but the
  results will not be verified by the autograder.

For debugging purposes, a single unit test can be run locally.  For example, you
can run the test case `1a-0-basic` using the following terminal command within
the `src/` subdirectory:
```
(XCS_ENV) $ python grader.py 1a-0-basic
```

## How to create a typeset submission using LaTeX
You are welcome to typeset your submission in any legible format (including
handwritten).  For those who are trying LaTeX for the first time, consider using
the following build process (we've tried to streamline it as much as possible
for you!).  All instructions that follow are for our build process, which will
require a working installation of [TeX Live](https://www.tug.org/texlive/) (This
website has installation instructions for Windows, Linux, and Mac).  Most linux
distributions come pre-loaded with this.  Mac users can download and install it
from [mactex](https://tug.org/mactex/)

All LaTeX files are in the `tex/` subdirectory. Your question responses will be
typed into the file `tex/submission.tex`.  We recommend attempting to compile
the document before typing anything in `tex/submission.tex`.

Run `make` form the root directory.  Complete `make` documentation is
provided within the Makefile.  To get started, clone the repository and try out
a simple `make` command:
```
$ make clean -s
```

If the command runs correctly, it will remove the assignment PDF from your root
directory.  Don't worry though!  Try recreating it again using the following
command:
```
$ make without_solutions -s
```

After some file shuffling and a few passes of the LaTeX compiler, you should see
a fresh new assignment handout in the root directory.  Now try the following
command:
```
$ make with_solutions -s
```

You should now see a `\*_Solutions.pdf` file in your root directory.  This
contains the content from the original handout as well as your solutions (those
typed into `tex/submission.tex`)!  If you haven't edited `tex/submission.tex`
yet, it will probably look a lot like the `without_solutions` version.

To see what it looks like with some solution code, open up `tex/submission.tex`
and enter the following code between the tags `### START CODE HERE ###` and
`### END CODE HERE ###`:
```latex
\begin{answer}
  % ### START CODE HERE ###
  \LaTeX
  % ### END CODE HERE ###
\end{answer}
```

Now run the following command:
```
$ make -s
```

This command re-runs the default `make` target, which is, conveniently,
`make with_solutions`.  Opening the file `\*_Solutions.pdf`, you should see
something like the following:

<img src="https://render.githubusercontent.com/render/math?math=\LaTeX">

## How to create a typeset submission using LaTeX on Overleaf
[Overleaf](https://www.overleaf.com/) is an online WYSIWYG editor.  While we
recommend becoming familiar with compiling LaTeX locally, you may instead prefer
the ease of Overleaf. Follow these steps to get set up with Overleaf (after
creating an account for yourself):

1. Create a new "Blank Project".
<img src="README_media/1.png">
2. Give the project a name.
3. Delete the file named "main.tex".
<img src="README_media/3.png">
4. Upload the following files to your project:
- "submission.tex"
- "macros.tex"
<img src="README_media/4.png">
5. Open the Overleaf menu at the top left.
<img src="README_media/5.png">
6. Change the "Main document" to "submission.tex".
<img src="README_media/6.png">
7. Recompile the document.
<img src="README_media/7.png">

Good luck with the assignment!  Remember that you can always submit organized
and legible handwritten PDFs instead of typeset documents.
