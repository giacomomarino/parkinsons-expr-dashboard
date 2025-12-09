# Parkinson's Gene Expression Dashboard

This is a dashboard for gene expression exploration in different tissues and cell types extracted from the [ARCHS4](https://archs4.org) resource which contains uniformly aligned RNA-seq samples extracted automatically from the Gene Expression Omnibus (GEO).

The data needs to be downloaded and placed in a ```data``` directory from [here](https://drive.google.com/file/d/13Q9ASCOeIgTCDvIE0P6Zl-s-z6dBQ-Rs/view?usp=drive_link) (~330 MB).

To run the dashboard in development:

```bash
# create virtual env
python3 -m venv .venv
# install requirements.
pip3 install -r requirements.txt

# run server (debug mode)
python3 server.py
```