# Forecast Data

> ### Extract Data
-  Get all symbols for cryptocurrency (from Binance) 
-  Extract cryptocurrency data per 1-day (From 01/01/2020 - Until 31/12/2023) (from Binance)

#

> ### Preperation Data
- Select a Target from cryptocurrency
- Drop big-null columns in data
- Drop holidays rows (saturday and sunday)
- Backrward and Forward fill by columns in dataframe
- Keep specific data (most famous)
- Return a correlation dataframe of Target

#
 
> ### Visualization
- Plot of missing columns of dataframe
  ![Screenshot 2024-11-12 194815](https://github.com/user-attachments/assets/9d798332-3a41-4e17-92b2-59d0ded976f1)

- Plot dataframe of correlation (2-correlation)
  ![Screenshot 2024-11-12 194924](https://github.com/user-attachments/assets/8a15674a-cecb-4c8a-9022-5067497826cb)

- Plot of Target value
 ![Screenshot 2024-11-12 194705](https://github.com/user-attachments/assets/52ef1c38-8c3f-4bf6-8e2b-d66b78e023b1)

- Animated dataframe with train test split
  ![plot_bars_lines](https://github.com/user-attachments/assets/3a534bdc-86e2-41bc-9caa-a9e10315d9d2)

  
- Recursive forecast of dataframe (Train 80% and +30) (cost function is RMSE and Residuals plot)
![Screenshot 2024-11-12 194720](https://github.com/user-attachments/assets/8756b881-10be-4163-93e7-c0769d3fb923)
![Screenshot 2024-11-12 194728](https://github.com/user-attachments/assets/de2fbd79-c116-4a09-bb33-3dca395473fa)
![Screenshot 2024-11-12 194738](https://github.com/user-attachments/assets/b686d0be-8f0c-4eb6-b102-39545372898b)
  ![Screenshot 2024-11-12 194748](https://github.com/user-attachments/assets/675ea813-de34-47c1-aac1-689f089ad0da)

#

> ### Forecast 
- Linear Regression Model (Regression)
