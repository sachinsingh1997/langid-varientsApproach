## Task-2 ##
- Our task was focussed on developing a model to distinguish between language variants. Here we wish to distinguish between European Portuguese (PT-PT) and Brazilian Portuguese (PT-BR).

## Data ##
- We had 2 datasets (pt-pt and pt-br) with 1.9Mn and 1.5Mn lines.
- Due to unability to process such a large file (not high spec system) at once hence we used random sampler and got 65,000 sampels from **pt-br** and 50,000 samples from **pt-pt** respectively. (**command: shuf -n N input_file > output_file**)
- We gave languaegs ID's as {pt-br: 0, pt-pt: 1} respectively.

----------------------------------------------------------------------

## Pre-Processing ##
- All the texts were converted to lower case.
- Removal of punctuations.
- All the digits were removed from the text sentences.
- Series of contiguous white spaces were replaced by single space.
- Removal of hyperlinks

#### Representation ####
- We used **TfidfVectorizer** for representing the text in our corpus.
- Keeping top 6000 words for reprsentation of the sentences.
-----------------------------------------------------------------------

## Data split for our model train and test##
- We splitted the data into 80/20 for trianing and testing at the same time keeping in mind to have similar number of instances for all the three langauges in test too.
- Total **pt-br: 47554** and **pt-pt: 50000**.
- **Train dataset**: {pt-br: 38052, pt-pt: 39991}
- **Test dataset**: {pt-br: 9502, pt-pt: 10009}

-----------------------------------------------------------------------

## ML Model ##
- Used sklearn for importing the models.
- We have used **LogisticRegression** algorithm for our classification with **solver='lbfgs'**.

------------------------------------------------------------------------

## Results ##

- Our system achieved an accuracy of **81.9%**.
- The confusion matrix for the result is as below:

    |     |  0  |  1  |
    |-----|--------|:------:|
    |  **0**  | 7735    |  1767   |
    |  **1**  |  1761   |   8248  |
    


- The Classification report is as below:

    |      | precision | recall | f1-score | support |
    | ---- |:---------:|:------:|:--------:|:-------:|
    |  **pt-br**  |     0.81  |    0.81|      0.81|      9502|
    |  **pt-pt**  |     0.82  |    0.82|      0.82|      10009|
    
   
   ------------------------------------------------------------------- 
    
## Running the script and results ##

- To run our model the command is as below:
- **python3 script_task2.py data.pt-br data.pt-pt langid-variants.test**
- In the above command 1st agument should be Brazilian Portuguese (pt-br) file, 2nd as European Portuguese (pt-pt) and 3rd as the testing file.
- **test_result.txt** will contain the language label predicted for the **langid-variants.test** file once ran on our model.
- test_results.txt above contains the output on the langid.tets file provided for us to test.
- Tags here are numerical **0: pt-br, 1: pt-pt**
---------------------------------------------------------------------
