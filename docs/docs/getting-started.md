Getting started
===============

## Environment and Dependencies
The repo uses poetry to handle the environment and dpendencies


## Immutable Data

The list of slides in the excel sheet format is used as the raw immutable data from which canonical data that is input the model is used. Name of the file does not matter as long as the file extension is .xlsx. Typically the excel sheet has two columns. Slide_id and Classification. The classification column is optional. This needs to be manually placed in the ./data/raw/ folder.