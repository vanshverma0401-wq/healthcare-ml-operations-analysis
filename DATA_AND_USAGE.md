# Data and usage

## Dataset provenance

The billing and occupancy workstreams use the synthetic
[Healthcare Dataset by Prasad Patil](https://www.kaggle.com/datasets/prasad22/healthcare-dataset).
The provider describes generation with Python's Faker library and states that
it contains no real patient information. The Kaggle metadata API lists
**CC0: Public Domain**, dataset version **2**, last updated May 8, 2024; these
fields were checked November 8, 2026.

The source has 55,500 rows. The V4 notebook's duplicate-removal step produces
54,966 rows. These counts refer to the supplied historical workflow, rather than
a new download or retraining run. Some scripts expect derived input columns
or earlier prepared filenames; a fresh provider download alone does not prove
that every retained experiment will run unchanged.

## Public content

This repository publishes reviewed coursework source, the complete V4 and V5
billing notebook cells with safe saved outputs, aggregate figures, and new
portfolio documentation. Original figure bytes are retained and fingerprinted.
Privacy transformations, path changes and removed outputs are recorded in
[the source manifest](technical/source_manifest.json).

Raw CSVs and record-level previews are excluded, including synthetic name-like
fields. Private cloud links, machine paths, account metadata, package-install
logs, course feedback, original archives and documents carrying confidentiality
markings are not public artifacts. No confidential manual or presentation is
rehosted or reproduced in full.

## Team attribution

The project was completed by **Hinsoukpo Dagan, Drew Fleming, Kathleen
Hinderscheid, Benjamin Philbin, Dymend Teklab and Vansh Verma**. The spelling
Dymend follows the supplied final billing presentation. Vansh's contribution
covers cleaning, feature engineering, preprocessing, model comparison,
evaluation, documentation and presentation within this collaborative work.
Original code and new publication/audit helpers are distinguished in the
technical inventories.

The dataset's CC0 status does not relicense every team document or third-party
library. No blanket software license is asserted for the complete archive.

## Interpretation

Synthetic encounters, generated diagnoses and artificial occupancy/capacity
proxies are educational inputs. They do not establish real-patient experience,
observed staffing capacity, clinical validation, a production hospital system,
or a privacy-compliance certification. Evaluation limitations and unresolved
reproduction issues remain explicit in [Methods](METHODS_AND_LIMITATIONS.md).
