# Forecast Data

> ### Extract Data
-  Get all symbols for cryptocurrency
-  Extract cryptocurrency data from Binance per 1-day (From 01/01/2020 - Until 31/12/2023)

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
  ![Screenshot 2024-11-12 194705](https://github.com/user-attachments/assets/c73f5875-7d08-4204-8ebe-25c8bdd64eaf)

- Animated dataframe with train test split
  ![Screenshot 2024-11-12 194801](https://github.com/user-attachments/assets/73283d06-4e4c-459a-a989-610a1a9412bf)
  
- Recursive forecast of dataframe (Train 80% and +30) (2 Plots)
  ![Screenshot 2024-11-12 194720](https://github.com/user-attachments/assets/f0977c6d-5759-41af-9703-cf8db25d715c)
  ![Screenshot 2024-11-12 194728](https://github.com/user-attachments/assets/72af60ed-009f-407c-a887-e33a94bdcfb7)
  ![Screenshot 2024-11-12 194738](https://github.com/user-attachments/assets/e324e307-6caa-45f7-80ea-315cb5ac4ca3)
  ![Screenshot 2024-11-12 194748](https://github.com/user-attachments/assets/675ea813-de34-47c1-aac1-689f089ad0da)

#

> ### Forecast 
- Linear Regression Model (Regression)
