# Development and validation of a quick, automated and reproducible ATR FT-IR spectroscopy machine-learning model for Klebsiella pneumoniae typing

This paper has been submitted for publication in Journal of Clinical Microbiology [https://journals.asm.org/journal/jcm](https://journals.asm.org/journal/jcm).

## Authors

Ângela Novais<sup>1,2</sup>, Ana Beatriz Gonçalves<sup>1,2</sup>, Teresa G. Ribeiro<sup>1,2,3</sup>, Ana R. Freitas<sup>1,2,45</sup> ,
Gema Méndez<sup>5</sup>, Luis Mancera<sup>5</sup>, Antónia Read<sup>6</sup>, Valquíria Alves<sup>6</sup>, Lorena López-
Cerero<sup>7,8</sup>, Jesús Rodríguez-Baño<sup>7,8</sup>, Álvaro Pascual<sup>7,8</sup>, Luísa Peixe<sup>1,2,3</sup>

<sup>1</sup> UCIBIO – Applied Molecular Biosciences Unit, Department of Biological Sciences, Faculty of
Pharmacy University of Porto, Porto, Portugal

<sup>2</sup> Associate Laboratory i4HB - Institute for Health and Bioeconomy, Faculty of Pharmacy,
 University of Porto, Porto, Portugal

<sup>3</sup> CCP – Culture Collection of Porto, Faculty of Pharmacy University of Porto, Porto, Portugal

<sup>4</sup> 1H-TOXRUN, One Health Toxicology Research Unit, University Institute of Health Sciences,
CESPU, CRL, Gandra, Portugal;

<sup>5</sup> Clover Bioanalytical Software, Granada, Spain;

<sup>6</sup> Clinical Microbiology Laboratory, Local Healthcare Unit, Matosinhos, Portugal;

<sup>7</sup> Unidad Clínica de Enfermedades Infecciosas y Microbiología. Hospital Universitario Vírgen
Macarena. Instituto de Biomedicina de Sevilla (IBIS; CSIC/Hospital Virgen
Macarena/Universidad de Sevilla). Sevilla. Spain.

<sup>8</sup> Departmentos de Microbiología y Medicina. Universidad de Sevilla. Spain.

## Abstract

**BACKGROUND**: The reliability of Fourier-Transform infrared spectroscopy (FT-IR) for
Klebsiella pneumoniae typing and outbreak control has been previously assessed, but issues
remain in standardization and reproducibility. We developed and validated a reproducible FT-IR
with attenuated total reflectance (ATR) workflow for identification of K. pneumoniae lineages.

**MATERIAL AND METHODS**: We used 293 isolates representing multidrug resistant K.
pneumoniae lineages causing outbreaks worldwide (2002-2021) to train a random forest
classification model based on capsular (KL)-types discrimination. This model was validated
with 280 contemporaneous isolates (2021-2022), using wzi sequencing and whole genome
sequencing as references. Repeatability and reproducibility were tested in different culture
media and instruments throughout time.

**RESULTS**: Our RF model allowed classification of 33 capsular (KL)-types and up to 36
clinically-relevant K. pneumoniae lineages based on the discrimination of specific KL- and O-
type combinations. We obtained high rates of accuracy (89%), sensitivity (88%) and specificity
(92%) including from cultures obtained from the clinical sample allowing to obtain typing
information the same day than bacterial identification. The workflow was reproducible in
different instruments throughout time (>98% correct predictions). Direct colony application,
spectral acquisition and automated KL prediction through Clover MS Data analysis software
allows a short time-to-result (5 min/isolate).

**CONCLUSIONS**: We demonstrated that FT-IR ATR spectroscopy provides meaningful,
reproducible and accurate information at a very early stage (as soon as bacterial identification)
to support infection control and public health surveillance. The high robustness together with
automated and flexible workflows for data analysis provide opportunities to consolidate real
time applications at a global level.

## Data

The spectra data used in this study is provided in the `data` folder.

**NOTE**: The data in this repository has been preprocessed as established on the manuscript:

> - Standard normal variate
> - Saviztky-Golay filter (window length: 9; polynomial order: 2; derivative order: 2)
> - Region selection: from 1200 cm-1 to 900 cm-1.

The raw profiles have been uploaded to the Clover Garden Repository, being available for public download in this link: [https://platform.clovermsdataanalysis.com/garden/collection/FTIR001](https://platform.clovermsdataanalysis.com/garden/collection/FTIR001)

## Reproducing the results

### Prerequisites

#### Python

Last tested with version 3.8.10 [https://www.python.org/downloads/release/python-380/]([21.1.1](https://www.python.org/downloads/release/python-380/)) Expected to work with newer python versions.

#### pip

Automatically installed together with Python. Last tested with version 21.1.1

#### Python venv module

Automatically installed together with Python.

### Installation

Clone this repository using the git cli or your preferred tool.

Open a command prompt terminal, navigate to the project's main folder and ensure that Python is installed and detected

```bash
> python -V
Python 3.8.10
```

Create a virtual environment using the `venv` module.

```bash
> python -m venv venv
```

Activate the virtual environment you have just created

```bash
>.\venv\Scripts\activate
```

Install required third-party libraries

```bash
(venv)>pip install -Ur requirements.txt
```

### Running the script

All source code used to generate the results in the paper is located in the `scripts` folder. The main script is the file `main.py` situated in the root folder

First ensure the virtual environment is activated

```bash
>.\venv\Scripts\activate
```

Then run the file `main.py`:

```bash
> python main.py
```

## License

All source code is made available under a BSD 3-clause license. You can freely
use and modify the code, without warranty, so long as you provide attribution
to the authors. See `LICENSE.md` for the full license text.

The manuscript text is not open source. The authors reserve the rights to the
article content, which is currently submitted for publication in the
ASM Journals.
