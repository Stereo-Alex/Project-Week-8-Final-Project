<img src="https://bit.ly/2VnXWr2" alt="Ironhack Logo" width="100"/>

# Staying informed without the work
*Alex jurado*

*Jan2021 Data Analytics *

## Content
- [Project Description](#project-description)
- [Dataset](#dataset)
- [Cleaning](#cleaning)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Conclusion](#conclusion)
- [Future Work](#future-work)
- [Workflow](#workflow)
- [Organization](#organization)
- [Links](#links)

## Project Description
Script that once executed will scrape a few newspapers and generate a summary and a visualization as well as email the user the results. 


## Dataset
* Due to the nature of the project the dataset will change everyday and the previous days one will be discarded. Down the line, a copy could be saved. 
* Each day six data frames are generated and kept separate to avoid contaminating each other.

## Cleaning
Cleaning was done using regex and removing all unwanted characters to facilitate the tokenization of the 6 separate dataframes.


## Model Training and Evaluation
There was no model training. We used a hugging face transformer named for a pre trained BART model 
* The model combines BERT and GPT3 for the encoder-decoder architecture. 

## Conclusion
* The main challenge was the pdf generation as well as understanding and using the pre trained model. 

## Future Work
Full automatisation by moving the notebook to a google colab notebook

## Workflow
Data collection and cleaning was the first step followed by modelling and finally the pdf and emailer generation

## Organization
Used Trello as well as a very structured Notebook to do all the work

Currently there is one notebook that will run once a day as well as folders to hold the word clouds. 

## Links



[Repository](https://github.com/Stereo-Alex/Project-Week-8-Final-Project)  
[Trello](https://trello.com/b/VvdxmLI6/final-project)  
