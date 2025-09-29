# Titanic Machine Learning Problem 
I have attempted to solve my first Machine Learning problem using the knowledge I gained in my Machine Learning class. I have started with a simple Decision Tree model, and progressed to Random Forest with bootstrapping and hyper parameter tuning, which is the version I am presenting here. 

## Table of Contents 
  - [Features](#features)
  - [Accuracy](#accuracy)
  - [Further Improvements](#further_improvements)

## Features 
1. Preprocessing:
   I have decided to not include Name, Ticket, Fair and Cabin columns.
   For the Age column, I have used mean imputation to fill in the missing data, as well as divide the ages into 4 groups for convience.
   
   <img width="549" height="64" alt="image" src="https://github.com/user-attachments/assets/f2fbe83b-dad1-49d6-a2b0-40786451bd02" />
   
   For the Embarked column, missing data was filled using the mode imputation. Furthermore, I used One-hot encoding to convert the categorical data.
   
   <img width="647" height="27" alt="image" src="https://github.com/user-attachments/assets/11393b00-3abe-4d0f-8ac9-aef27ad42f1e" />
   <img width="753" height="94" alt="image" src="https://github.com/user-attachments/assets/63e6ea34-1aa3-4859-b310-eabf35bac2f7" />
   
   Same method was used for the Sex column, converting male to 1.0 and female to 0.0:
   <img width="702" height="90" alt="image" src="https://github.com/user-attachments/assets/3f480362-44b8-43fb-90a1-f07e1493105f" />
2. Parameter Tuning:
   The initial Random Forest model has 5 fixed parameters:
   
   <img width="411" height="119" alt="image" src="https://github.com/user-attachments/assets/4ec02e3e-172c-4548-a7c9-b91f999e1610" />
   
   As well as 5 parameters we want to tune:
   
   <img width="462" height="110" alt="image" src="https://github.com/user-attachments/assets/efc61dca-3099-4286-852e-6b14df208e8f" />
   
   Using RandomizedSearchCV, we do 30 iterations with cv=5 to find the best value for each parameter.
   
   <img width="296" height="165" alt="image" src="https://github.com/user-attachments/assets/77056482-8af5-4aa2-9be0-b9277a4be0aa" />
   
   At the end, the parameters, as well as the Mean, Std and OOB are printed out for debug, and finally, the test and training scores.
3. Other:
   I also implemented a timer to record how long it takes to tune the parameters. The average tuning time lies between 25 and 30 seconds.
   
## Accuracy 
Program execusion yields similar results to this example: 
<img width="306" height="181" alt="image" src="https://github.com/user-attachments/assets/878710f1-d80f-49ff-bb43-e94064703ea1" /> 
- The usual Mean and OOB is 0.80 ~ 0.82, Std is around 0.01-0.02
- Training accuracy is around 84 - 85%
- Test accuracy is 88 - 90%
- The Keggle score is 0.77751
  <img width="898" height="102" alt="image" src="https://github.com/user-attachments/assets/26078634-69a1-4e47-9803-9e9b620900cd" />
  
## Further_Improvements
Going further, I am planning to work on the Fare column to increase the accuracy, as well as try out feature engineering.
I definitely want to try to implement other Bagging approaches, as well as Boosting, such as Adaboost, various Gradient boosting and so on.
