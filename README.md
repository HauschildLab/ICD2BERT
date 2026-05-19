# ICBERT

## Getting Started

Create a Python virtual environment and install the packages from the [requirements.txt](./requirements.txt) file

```bash
python -m venv venv
source ./venv/bin/activate
pip install -r requirements.txt
```

Download the MIMIC-IV dataset from [PysioNet](https://physionet.org/content/mimiciv/2.2/). The paper uses version 2.2 of the dataset.
Insert the dataset into the mimic-iv folder at [./data/mimic-iv](./data/mimic-iv/)

The full dataset is not required to reproduce the results from the paper but only the following files:

* [admissions.csv.gz](https://physionet.org/content/mimiciv/2.2/hosp/admissions.csv.gz)
* [diagnoses_icd.csv.gz](https://physionet.org/content/mimiciv/2.2/hosp/diagnoses_icd.csv.gz)
* [procedures_icd.csv.gz](https://physionet.org/content/mimiciv/2.2/hosp/procedures_icd.csv.gz)
* [d_icd_diagnoses.csv.gz](https://physionet.org/content/mimiciv/2.2/hosp/d_icd_diagnoses.csv.gz)
* [d_icd_procedures.csv.gz](https://physionet.org/content/mimiciv/2.2/hosp/d_icd_procedures.csv.gz)

These files should be placed in [./data/mimic-iv/hosp](./data/mimic-iv/hosp/)


Download the EHRSHOT dataset and person.csv table from [redivis](https://stanford.redivis.com/datasets/53gc-8rhx41kgt).
Insert EHRSHOT_ASSETS into the [./data](./data/) folder and person.csv into that folder [./data/EHRSHOT_ASSETS](./data/EHRSHOT_ASSETS)

Finally run numbered scripts in [./src](./src/) in order.