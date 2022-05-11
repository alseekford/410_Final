# 410 Final Project
### Anne Louise Seekford and Lisa Fukutoku
#### DATA 410 - Advanced Applied Machine Learning - Multiple Boosting and LightGBM
#### 05.11.22

Created and implemented our own multiple boosting algorithm to combinations of different regressors on the "Major Social Media Stock Prices" dataset. Additional application of LightGBM algorithm. 

## Overview

For the final project, we created and applied a multiple boosting algorithm of our creation to five regression techniques on the Major Social Media Stock Price Dataset. This dataset is multivariate and our aim is to answer questions relevant to the peak and trough of buying and selling stocks. 


## Data

The Major Social Media Stock Price dataset (Kanawattanachai, 2022), retrieved from Kaggle, consists of stock price details from five dominant social media platforms: Facebook, Twitter, Snapchat, Etsy, and Pinterest. The stock symbol, adjusted-close price, open and close price, high and low price, trading volume and date of the stock are described for each platform, per period. We will delve into exploring the highest and lowest price at which a stock traded during a period - as these numbers indicate the financial health and stability of the company. We plan to cross reference our results with current events to see if a real-life explanation for the results exists.  

Snippet of Dataset:  

<p align = 'center'><img width="470" alt="Screen Shot 2022-05-11 at 2 12 21 PM" src="https://user-images.githubusercontent.com/71660299/167917862-8c75abc7-d37c-45ab-af44-ebbaf37ac273.png">  



## Analysis Methods and Preprocessing

For the data preprocessing, all data with missing values were removed. Then, we set y as the trading volume column with our feature variable, x, as the high and low stock price. Due to the excessive runtime and massive dataset, we had to subset the features into ninths, each containing roughly 1,000 rows each. We then created a function to call for each x0, x1, x2, …, x8 to make our code cleaner and easier to run and read. As the analytical/machine learning methods, we first created and applied a multiple boosting algorithm of our creation to two regression techniques. By utilizing Locally Weighted Linear Regression, Boosted LOWESS, Random Forest, XGBoost, and our own “super booster” regressors, we were able to compare MSE results to LightGBM. 


## Multiple Boosting Algorithm

  
<p align = 'center'><img width="365" alt="Screen Shot 2022-05-11 at 2 13 22 PM" src="https://user-images.githubusercontent.com/71660299/167918024-8d29e7dc-6386-4ac2-b80f-6f59c21046b3.png">  


Our multiple booster was then ran in a K-Fold Cross Validation Loop, along with other regressors.  

<p align = 'center'><img width="739" alt="Screen Shot 2022-05-11 at 2 11 52 PM" src="https://user-images.githubusercontent.com/71660299/167917784-292ec95a-61fb-4d08-be3e-acbf8b708187.png">  


  #### Results  
  
  Due to the size of the dataset, we had to subset.
  
<p align = 'center'> <img width="417" alt="Screen Shot 2022-05-11 at 2 29 08 PM" src="https://user-images.githubusercontent.com/71660299/167920646-3d9bed27-d4e3-4d28-9f72-4183b7f49e9e.png">  
  
  After taking the average of all nine subsets' MSE, we came to these results: 
  
  <p align = 'center'> <img width="302" alt="Screen Shot 2022-05-11 at 2 31 56 PM" src="https://user-images.githubusercontent.com/71660299/167921069-207b35d3-f2c7-4c75-862b-b24bed7dc29c.png">

  


## LightGBM

LightGBM is a gradient-boosting framework that utilizes a vertically-based tree structure learning algorithm. Based on decision tree algorithm, it is used for ranking, regression, classification, and many other machine learning tasks. Due to the vertical flow, LightGBM can significantly outperform XGBoost and other regressors in terms of both computational speed and memory consumption (Guolin et al.). Therefore, when growing on the same leaf in Light GBM, the leaf-wise algorithm can reduce more loss than the level-wise algorithm and hence results in much better accuracy which can rarely be achieved by any of the existing boosting algorithms. With that being said however, LightGBM is sensitive to overfitting, and is recommended to be used on larger datasets. There is a variety of parameters to include to improve results, a few examples including (Mandot, 2018):
  - learning_rate: determines the impact each tree has on the final outcome
  - num_leaves: number of leaves in a full tree (31 is default)
  - num_boost_round: number of boosting iterations
  - boosting: defines the type of algorithm running (can choose traditional Gradient Boosting Decision Tree (gbdt), Random Forest (rf), etc.)
  

Leaf-wise tree growth in LightGBM:  

<p align = 'center'><img width="838" alt="Screen Shot 2022-05-11 at 2 25 01 PM" src="https://user-images.githubusercontent.com/71660299/167919974-94b6d4dd-e80d-49ac-b2db-6ab9eefb719c.png">  
  
LightGBM is called “Light” because of its computation power and giving results faster. It takes less memory to run and is able to deal with large volumes of data having more than 10,000+ rows, especially when one needs to achieve a high accuracy of results. Due to LightGBM's extreme computational power, we were able to run the dataset complete, without subsetting like we had done to the prior algorithms.  
  
<p align = 'center'> <img width="625" alt="Screen Shot 2022-05-11 at 2 27 06 PM" src="https://user-images.githubusercontent.com/71660299/167920277-9cca735c-8169-4d1d-beb7-61e599bbf32c.png">  
  
  #### Results:  
<p align = 'center'> <img width="257" alt="Screen Shot 2022-05-11 at 2 28 17 PM" src="https://user-images.githubusercontent.com/71660299/167920464-02b1aebb-76e3-487a-bb53-08e624c67009.png">  

  



## References

Guolin, et al. (n.d.). LightGBM: A highly efficient gradient boosting decision tree. Retrieved March 10, 2022, from https://papers.nips.cc/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf
Hartman, D. (2017, February 7). How do stock prices indicate financial health? Finance. Retrieved April 15, 2022, from https://finance.zacks.com/stock-prices-indicate-financial-health-9096.html 
Mandot, P. (2018, December 1). What is LIGHTGBM, how to implement it? how to fine tune the parameters? Medium. Retrieved March 10, 2022, from https://medium.com/@pushkarmandot/https-medium-com-pushkarmandot-what-is-lightgbm-how-to-implement-it-how-to-fine-tune-the-parameters-60347819b7fc
Prasert Kanawattanachai. (April 2022). Major social media stock prices 2012-2022, Version 1. Retrieved April 14, 2022 from https://www.kaggle.com/datasets/prasertk/major-social-media-stock-prices-20122022.
 


## Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://support.github.com/contact) and we’ll help you sort it out.
