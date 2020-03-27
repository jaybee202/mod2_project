# King County Housing Prices
This README.md lists the github link, project members, goals, responsibilities, and a summary of the files in the repository.


## Github link
All project documents are available here:
https://github.com/jaybee202/mod2_project


## Project File Summary
The Github repository includes the following files and folders

- README.md - a summary repository content
- /archived_files - includes Jupyter Notebooks of code developed but not ultimately used
- /data - all data files pertaining to King County housing prices processed data files
- /presentation - PowerPoint and pdf files of the final presentation
- /project_documents - Onboarding project documents provided at the start of this project
- /research - a map of King County zip codes
- /step1_data_cleaning - a Jupyter Notebook (JNB) used to clean the original data
- /step2_feature_correlation_analysis - descriptive statics analysis of the cleaned data
- /step3_train_test_split - JNB used to split and save the train and test data
- /step4_model_and_feature_selection - Files used to run OLS linear regression models and feature removal
- /step5_model_testing - JNB testing the test data against the model
- /step6_analysis - JNB used to acquire specific data points needed for the final presentation


## Project Members
    Joe Buzzelli


## Project Scenario
For the second module in Flatiron's Data Science bootcamp provided me with data pertaining to King County home prices.  The project required the creation of a model to predict a house's price.  In order to scope the effort and provide direction, I assumed a fictitious scenario.  The project was framed as a pilot project in support of Redfin Now, an online real estate service where an individual can sell their house to Redfin.

Given the project scenario, I created a linear regression based on sixteen data points pertaining to King County home prices.  Of these sixteen data points, one was treated as a categorical variable, the being the zip code.  The final model yielded an adjusted r2 value of 0.87.

### Project objectives:
- Create a predictive model for housing prices in King County
- Reduce the quantity of features required by the model
- Apply project findings to a briefing to Redfin Now


## Methodology
__Step 1__: Clean the data my imputing null values and setting proper datatypes

__Step 2__: Review descriptive statistics of the data looking for highly correlated independent variables and identify categorical variables

__Step 3__: Split the data into train and test sets after one hot encoding the zip code data

__Step 4.1__: Feature selection conducted by looping OLS models and removing the highest p-value until none exceeded 0.05

__Step 4.2__: After reviewing the y_train data, applied a log function given its shape, applied a linear regression model with adjusted r2 equaling 0.87

__Step 5__: Fed the test data through the train model and with the resulting adjusted r2 roughly equal to the train r2 adjusted value

__Step 6__: Analyzed the data set for specific elements used in the final presentation 


## Project Responsibilities
Joe Buzzelli was responsible for all elements of this project
