# credit_card_fraud_analysis
 
original source: https://www.kaggle.com/hellbuoy/credit-card-fraud-detection


# Step 1: Reading and Understanding the Data



```python
# import all the required libraries and dependencies for dataframe
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime, timedelta

# import all libraries and dependencies for data visualization
pd.options.display.float_format='{:.4f}'.format
plt.rcParams['figure.figsize'] = [8,8]
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_colwidth', -1) 
sns.set(style='darkgrid')
import matplotlib.ticker as plticker
%matplotlib inline


# import all the required libraries and dependencies for machine learning

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics
import statsmodels.api as sm
import pickle
import gc 
from sklearn import svm




```

# Locate data


```python
# Local file path.

path = 'C:/Users/uddin/OneDrive/Desktop/Educations/data_analysis/creditcard.csv'
```

# Reading data using pandas


```python
# importing data 

df_card = pd.read_csv(path)

```

# Exploring data frames


```python
# list first 5 records 
df_card.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Time</th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
      <th>V8</th>
      <th>V9</th>
      <th>V10</th>
      <th>V11</th>
      <th>V12</th>
      <th>V13</th>
      <th>V14</th>
      <th>V15</th>
      <th>V16</th>
      <th>V17</th>
      <th>V18</th>
      <th>V19</th>
      <th>V20</th>
      <th>V21</th>
      <th>V22</th>
      <th>V23</th>
      <th>V24</th>
      <th>V25</th>
      <th>V26</th>
      <th>V27</th>
      <th>V28</th>
      <th>Amount</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0000</td>
      <td>-1.3598</td>
      <td>-0.0728</td>
      <td>2.5363</td>
      <td>1.3782</td>
      <td>-0.3383</td>
      <td>0.4624</td>
      <td>0.2396</td>
      <td>0.0987</td>
      <td>0.3638</td>
      <td>0.0908</td>
      <td>-0.5516</td>
      <td>-0.6178</td>
      <td>-0.9914</td>
      <td>-0.3112</td>
      <td>1.4682</td>
      <td>-0.4704</td>
      <td>0.2080</td>
      <td>0.0258</td>
      <td>0.4040</td>
      <td>0.2514</td>
      <td>-0.0183</td>
      <td>0.2778</td>
      <td>-0.1105</td>
      <td>0.0669</td>
      <td>0.1285</td>
      <td>-0.1891</td>
      <td>0.1336</td>
      <td>-0.0211</td>
      <td>149.6200</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0000</td>
      <td>1.1919</td>
      <td>0.2662</td>
      <td>0.1665</td>
      <td>0.4482</td>
      <td>0.0600</td>
      <td>-0.0824</td>
      <td>-0.0788</td>
      <td>0.0851</td>
      <td>-0.2554</td>
      <td>-0.1670</td>
      <td>1.6127</td>
      <td>1.0652</td>
      <td>0.4891</td>
      <td>-0.1438</td>
      <td>0.6356</td>
      <td>0.4639</td>
      <td>-0.1148</td>
      <td>-0.1834</td>
      <td>-0.1458</td>
      <td>-0.0691</td>
      <td>-0.2258</td>
      <td>-0.6387</td>
      <td>0.1013</td>
      <td>-0.3398</td>
      <td>0.1672</td>
      <td>0.1259</td>
      <td>-0.0090</td>
      <td>0.0147</td>
      <td>2.6900</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0000</td>
      <td>-1.3584</td>
      <td>-1.3402</td>
      <td>1.7732</td>
      <td>0.3798</td>
      <td>-0.5032</td>
      <td>1.8005</td>
      <td>0.7915</td>
      <td>0.2477</td>
      <td>-1.5147</td>
      <td>0.2076</td>
      <td>0.6245</td>
      <td>0.0661</td>
      <td>0.7173</td>
      <td>-0.1659</td>
      <td>2.3459</td>
      <td>-2.8901</td>
      <td>1.1100</td>
      <td>-0.1214</td>
      <td>-2.2619</td>
      <td>0.5250</td>
      <td>0.2480</td>
      <td>0.7717</td>
      <td>0.9094</td>
      <td>-0.6893</td>
      <td>-0.3276</td>
      <td>-0.1391</td>
      <td>-0.0554</td>
      <td>-0.0598</td>
      <td>378.6600</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0000</td>
      <td>-0.9663</td>
      <td>-0.1852</td>
      <td>1.7930</td>
      <td>-0.8633</td>
      <td>-0.0103</td>
      <td>1.2472</td>
      <td>0.2376</td>
      <td>0.3774</td>
      <td>-1.3870</td>
      <td>-0.0550</td>
      <td>-0.2265</td>
      <td>0.1782</td>
      <td>0.5078</td>
      <td>-0.2879</td>
      <td>-0.6314</td>
      <td>-1.0596</td>
      <td>-0.6841</td>
      <td>1.9658</td>
      <td>-1.2326</td>
      <td>-0.2080</td>
      <td>-0.1083</td>
      <td>0.0053</td>
      <td>-0.1903</td>
      <td>-1.1756</td>
      <td>0.6474</td>
      <td>-0.2219</td>
      <td>0.0627</td>
      <td>0.0615</td>
      <td>123.5000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.0000</td>
      <td>-1.1582</td>
      <td>0.8777</td>
      <td>1.5487</td>
      <td>0.4030</td>
      <td>-0.4072</td>
      <td>0.0959</td>
      <td>0.5929</td>
      <td>-0.2705</td>
      <td>0.8177</td>
      <td>0.7531</td>
      <td>-0.8228</td>
      <td>0.5382</td>
      <td>1.3459</td>
      <td>-1.1197</td>
      <td>0.1751</td>
      <td>-0.4514</td>
      <td>-0.2370</td>
      <td>-0.0382</td>
      <td>0.8035</td>
      <td>0.4085</td>
      <td>-0.0094</td>
      <td>0.7983</td>
      <td>-0.1375</td>
      <td>0.1413</td>
      <td>-0.2060</td>
      <td>0.5023</td>
      <td>0.2194</td>
      <td>0.2152</td>
      <td>69.9900</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Get information about each column
df_card.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 284807 entries, 0 to 284806
    Data columns (total 31 columns):
     #   Column  Non-Null Count   Dtype  
    ---  ------  --------------   -----  
     0   Time    284807 non-null  float64
     1   V1      284807 non-null  float64
     2   V2      284807 non-null  float64
     3   V3      284807 non-null  float64
     4   V4      284807 non-null  float64
     5   V5      284807 non-null  float64
     6   V6      284807 non-null  float64
     7   V7      284807 non-null  float64
     8   V8      284807 non-null  float64
     9   V9      284807 non-null  float64
     10  V10     284807 non-null  float64
     11  V11     284807 non-null  float64
     12  V12     284807 non-null  float64
     13  V13     284807 non-null  float64
     14  V14     284807 non-null  float64
     15  V15     284807 non-null  float64
     16  V16     284807 non-null  float64
     17  V17     284807 non-null  float64
     18  V18     284807 non-null  float64
     19  V19     284807 non-null  float64
     20  V20     284807 non-null  float64
     21  V21     284807 non-null  float64
     22  V22     284807 non-null  float64
     23  V23     284807 non-null  float64
     24  V24     284807 non-null  float64
     25  V25     284807 non-null  float64
     26  V26     284807 non-null  float64
     27  V27     284807 non-null  float64
     28  V28     284807 non-null  float64
     29  Amount  284807 non-null  float64
     30  Class   284807 non-null  int64  
    dtypes: float64(30), int64(1)
    memory usage: 67.4 MB
    


```python
# data size 
df_card.shape
```




    (284807, 31)



 The dataset has 284807 rows and 31 columns 


```python
# data description 
df_card.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Time</th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
      <th>V8</th>
      <th>V9</th>
      <th>V10</th>
      <th>V11</th>
      <th>V12</th>
      <th>V13</th>
      <th>V14</th>
      <th>V15</th>
      <th>V16</th>
      <th>V17</th>
      <th>V18</th>
      <th>V19</th>
      <th>V20</th>
      <th>V21</th>
      <th>V22</th>
      <th>V23</th>
      <th>V24</th>
      <th>V25</th>
      <th>V26</th>
      <th>V27</th>
      <th>V28</th>
      <th>Amount</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>284807.0000</td>
      <td>284807.0000</td>
      <td>284807.0000</td>
      <td>284807.0000</td>
      <td>284807.0000</td>
      <td>284807.0000</td>
      <td>284807.0000</td>
      <td>284807.0000</td>
      <td>284807.0000</td>
      <td>284807.0000</td>
      <td>284807.0000</td>
      <td>284807.0000</td>
      <td>284807.0000</td>
      <td>284807.0000</td>
      <td>284807.0000</td>
      <td>284807.0000</td>
      <td>284807.0000</td>
      <td>284807.0000</td>
      <td>284807.0000</td>
      <td>284807.0000</td>
      <td>284807.0000</td>
      <td>284807.0000</td>
      <td>284807.0000</td>
      <td>284807.0000</td>
      <td>284807.0000</td>
      <td>284807.0000</td>
      <td>284807.0000</td>
      <td>284807.0000</td>
      <td>284807.0000</td>
      <td>284807.0000</td>
      <td>284807.0000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>94813.8596</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>-0.0000</td>
      <td>0.0000</td>
      <td>-0.0000</td>
      <td>0.0000</td>
      <td>-0.0000</td>
      <td>-0.0000</td>
      <td>-0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>-0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>-0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>-0.0000</td>
      <td>-0.0000</td>
      <td>88.3496</td>
      <td>0.0017</td>
    </tr>
    <tr>
      <th>std</th>
      <td>47488.1460</td>
      <td>1.9587</td>
      <td>1.6513</td>
      <td>1.5163</td>
      <td>1.4159</td>
      <td>1.3802</td>
      <td>1.3323</td>
      <td>1.2371</td>
      <td>1.1944</td>
      <td>1.0986</td>
      <td>1.0888</td>
      <td>1.0207</td>
      <td>0.9992</td>
      <td>0.9953</td>
      <td>0.9586</td>
      <td>0.9153</td>
      <td>0.8763</td>
      <td>0.8493</td>
      <td>0.8382</td>
      <td>0.8140</td>
      <td>0.7709</td>
      <td>0.7345</td>
      <td>0.7257</td>
      <td>0.6245</td>
      <td>0.6056</td>
      <td>0.5213</td>
      <td>0.4822</td>
      <td>0.4036</td>
      <td>0.3301</td>
      <td>250.1201</td>
      <td>0.0415</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.0000</td>
      <td>-56.4075</td>
      <td>-72.7157</td>
      <td>-48.3256</td>
      <td>-5.6832</td>
      <td>-113.7433</td>
      <td>-26.1605</td>
      <td>-43.5572</td>
      <td>-73.2167</td>
      <td>-13.4341</td>
      <td>-24.5883</td>
      <td>-4.7975</td>
      <td>-18.6837</td>
      <td>-5.7919</td>
      <td>-19.2143</td>
      <td>-4.4989</td>
      <td>-14.1299</td>
      <td>-25.1628</td>
      <td>-9.4987</td>
      <td>-7.2135</td>
      <td>-54.4977</td>
      <td>-34.8304</td>
      <td>-10.9331</td>
      <td>-44.8077</td>
      <td>-2.8366</td>
      <td>-10.2954</td>
      <td>-2.6046</td>
      <td>-22.5657</td>
      <td>-15.4301</td>
      <td>0.0000</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>54201.5000</td>
      <td>-0.9204</td>
      <td>-0.5985</td>
      <td>-0.8904</td>
      <td>-0.8486</td>
      <td>-0.6916</td>
      <td>-0.7683</td>
      <td>-0.5541</td>
      <td>-0.2086</td>
      <td>-0.6431</td>
      <td>-0.5354</td>
      <td>-0.7625</td>
      <td>-0.4056</td>
      <td>-0.6485</td>
      <td>-0.4256</td>
      <td>-0.5829</td>
      <td>-0.4680</td>
      <td>-0.4837</td>
      <td>-0.4988</td>
      <td>-0.4563</td>
      <td>-0.2117</td>
      <td>-0.2284</td>
      <td>-0.5424</td>
      <td>-0.1618</td>
      <td>-0.3546</td>
      <td>-0.3171</td>
      <td>-0.3270</td>
      <td>-0.0708</td>
      <td>-0.0530</td>
      <td>5.6000</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>84692.0000</td>
      <td>0.0181</td>
      <td>0.0655</td>
      <td>0.1798</td>
      <td>-0.0198</td>
      <td>-0.0543</td>
      <td>-0.2742</td>
      <td>0.0401</td>
      <td>0.0224</td>
      <td>-0.0514</td>
      <td>-0.0929</td>
      <td>-0.0328</td>
      <td>0.1400</td>
      <td>-0.0136</td>
      <td>0.0506</td>
      <td>0.0481</td>
      <td>0.0664</td>
      <td>-0.0657</td>
      <td>-0.0036</td>
      <td>0.0037</td>
      <td>-0.0625</td>
      <td>-0.0295</td>
      <td>0.0068</td>
      <td>-0.0112</td>
      <td>0.0410</td>
      <td>0.0166</td>
      <td>-0.0521</td>
      <td>0.0013</td>
      <td>0.0112</td>
      <td>22.0000</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>139320.5000</td>
      <td>1.3156</td>
      <td>0.8037</td>
      <td>1.0272</td>
      <td>0.7433</td>
      <td>0.6119</td>
      <td>0.3986</td>
      <td>0.5704</td>
      <td>0.3273</td>
      <td>0.5971</td>
      <td>0.4539</td>
      <td>0.7396</td>
      <td>0.6182</td>
      <td>0.6625</td>
      <td>0.4931</td>
      <td>0.6488</td>
      <td>0.5233</td>
      <td>0.3997</td>
      <td>0.5008</td>
      <td>0.4589</td>
      <td>0.1330</td>
      <td>0.1864</td>
      <td>0.5286</td>
      <td>0.1476</td>
      <td>0.4395</td>
      <td>0.3507</td>
      <td>0.2410</td>
      <td>0.0910</td>
      <td>0.0783</td>
      <td>77.1650</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>172792.0000</td>
      <td>2.4549</td>
      <td>22.0577</td>
      <td>9.3826</td>
      <td>16.8753</td>
      <td>34.8017</td>
      <td>73.3016</td>
      <td>120.5895</td>
      <td>20.0072</td>
      <td>15.5950</td>
      <td>23.7451</td>
      <td>12.0189</td>
      <td>7.8484</td>
      <td>7.1269</td>
      <td>10.5268</td>
      <td>8.8777</td>
      <td>17.3151</td>
      <td>9.2535</td>
      <td>5.0411</td>
      <td>5.5920</td>
      <td>39.4209</td>
      <td>27.2028</td>
      <td>10.5031</td>
      <td>22.5284</td>
      <td>4.5845</td>
      <td>7.5196</td>
      <td>3.5173</td>
      <td>31.6122</td>
      <td>33.8478</td>
      <td>25691.1600</td>
      <td>1.0000</td>
    </tr>
  </tbody>
</table>
</div>



# Step 2 : Data Cleaning

## check if there is any missing data, inconsistent data, duplicate data in the dataset 

1. Missing data


```python
# missing value check 
# show missing values in each column 
df_card.isnull()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Time</th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
      <th>V8</th>
      <th>V9</th>
      <th>V10</th>
      <th>V11</th>
      <th>V12</th>
      <th>V13</th>
      <th>V14</th>
      <th>V15</th>
      <th>V16</th>
      <th>V17</th>
      <th>V18</th>
      <th>V19</th>
      <th>V20</th>
      <th>V21</th>
      <th>V22</th>
      <th>V23</th>
      <th>V24</th>
      <th>V25</th>
      <th>V26</th>
      <th>V27</th>
      <th>V28</th>
      <th>Amount</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>284802</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>284803</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>284804</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>284805</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>284806</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>284807 rows × 31 columns</p>
</div>




```python
# show total missing values in the total dataframe 
df_card.isnull().sum().sum()
```




    0




```python
# Select the rows that have at least one missing value
df_card[df_card.isnull().any(axis=1)].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Time</th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
      <th>V8</th>
      <th>V9</th>
      <th>V10</th>
      <th>V11</th>
      <th>V12</th>
      <th>V13</th>
      <th>V14</th>
      <th>V15</th>
      <th>V16</th>
      <th>V17</th>
      <th>V18</th>
      <th>V19</th>
      <th>V20</th>
      <th>V21</th>
      <th>V22</th>
      <th>V23</th>
      <th>V24</th>
      <th>V25</th>
      <th>V26</th>
      <th>V27</th>
      <th>V28</th>
      <th>Amount</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>




```python
# alternative way: columnwise 
df_card.isna().any()
```




    Time      False
    V1        False
    V2        False
    V3        False
    V4        False
    V5        False
    V6        False
    V7        False
    V8        False
    V9        False
    V10       False
    V11       False
    V12       False
    V13       False
    V14       False
    V15       False
    V16       False
    V17       False
    V18       False
    V19       False
    V20       False
    V21       False
    V22       False
    V23       False
    V24       False
    V25       False
    V26       False
    V27       False
    V28       False
    Amount    False
    Class     False
    dtype: bool




```python
# sum of null values in each column 
df_card.isna().sum()
```




    Time      0
    V1        0
    V2        0
    V3        0
    V4        0
    V5        0
    V6        0
    V7        0
    V8        0
    V9        0
    V10       0
    V11       0
    V12       0
    V13       0
    V14       0
    V15       0
    V16       0
    V17       0
    V18       0
    V19       0
    V20       0
    V21       0
    V22       0
    V23       0
    V24       0
    V25       0
    V26       0
    V27       0
    V28       0
    Amount    0
    Class     0
    dtype: int64




```python
# This function gives output in a single value if any null is present or not.
df_card.isna().any().sum()
```




    0



looks like there is no missing values in the dataset. So there is no need to do futher work on missing values. In case we have null values in the data set, we need to fill those fields with some values so that we can know what values are there in those fields using fillna() function

2. Duplicate data: duplicate data are not needed at all in our data set. It affects the accuracy and efficiency of the analysis result. So, any duplicated data must be removed from the dataframe 


```python
# check data duplication 
df_card.duplicated()
```




    True




```python
# show duplicated rows 
df_card[df_card.duplicated()]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Time</th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
      <th>V8</th>
      <th>V9</th>
      <th>V10</th>
      <th>V11</th>
      <th>V12</th>
      <th>V13</th>
      <th>V14</th>
      <th>V15</th>
      <th>V16</th>
      <th>V17</th>
      <th>V18</th>
      <th>V19</th>
      <th>V20</th>
      <th>V21</th>
      <th>V22</th>
      <th>V23</th>
      <th>V24</th>
      <th>V25</th>
      <th>V26</th>
      <th>V27</th>
      <th>V28</th>
      <th>Amount</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>33</th>
      <td>26.0000</td>
      <td>-0.5299</td>
      <td>0.8739</td>
      <td>1.3472</td>
      <td>0.1455</td>
      <td>0.4142</td>
      <td>0.1002</td>
      <td>0.7112</td>
      <td>0.1761</td>
      <td>-0.2867</td>
      <td>-0.4847</td>
      <td>0.8725</td>
      <td>0.8516</td>
      <td>-0.5717</td>
      <td>0.1010</td>
      <td>-1.5198</td>
      <td>-0.2844</td>
      <td>-0.3105</td>
      <td>-0.4042</td>
      <td>-0.8234</td>
      <td>-0.2903</td>
      <td>0.0469</td>
      <td>0.2081</td>
      <td>-0.1855</td>
      <td>0.0010</td>
      <td>0.0988</td>
      <td>-0.5529</td>
      <td>-0.0733</td>
      <td>0.0233</td>
      <td>6.1400</td>
      <td>0</td>
    </tr>
    <tr>
      <th>35</th>
      <td>26.0000</td>
      <td>-0.5354</td>
      <td>0.8653</td>
      <td>1.3511</td>
      <td>0.1476</td>
      <td>0.4337</td>
      <td>0.0870</td>
      <td>0.6930</td>
      <td>0.1797</td>
      <td>-0.2856</td>
      <td>-0.4825</td>
      <td>0.8718</td>
      <td>0.8534</td>
      <td>-0.5718</td>
      <td>0.1023</td>
      <td>-1.5200</td>
      <td>-0.2859</td>
      <td>-0.3096</td>
      <td>-0.4039</td>
      <td>-0.8237</td>
      <td>-0.2833</td>
      <td>0.0495</td>
      <td>0.2065</td>
      <td>-0.1871</td>
      <td>0.0008</td>
      <td>0.0981</td>
      <td>-0.5535</td>
      <td>-0.0783</td>
      <td>0.0254</td>
      <td>1.7700</td>
      <td>0</td>
    </tr>
    <tr>
      <th>113</th>
      <td>74.0000</td>
      <td>1.0384</td>
      <td>0.1275</td>
      <td>0.1845</td>
      <td>1.1099</td>
      <td>0.4417</td>
      <td>0.9453</td>
      <td>-0.0367</td>
      <td>0.3510</td>
      <td>0.1189</td>
      <td>-0.2433</td>
      <td>0.5781</td>
      <td>0.6747</td>
      <td>-0.5342</td>
      <td>0.4466</td>
      <td>1.1229</td>
      <td>-1.7680</td>
      <td>1.2412</td>
      <td>-2.4495</td>
      <td>-1.7473</td>
      <td>-0.3355</td>
      <td>0.1025</td>
      <td>0.6051</td>
      <td>0.0231</td>
      <td>-0.6265</td>
      <td>0.4791</td>
      <td>-0.1669</td>
      <td>0.0812</td>
      <td>0.0012</td>
      <td>1.1800</td>
      <td>0</td>
    </tr>
    <tr>
      <th>114</th>
      <td>74.0000</td>
      <td>1.0384</td>
      <td>0.1275</td>
      <td>0.1845</td>
      <td>1.1099</td>
      <td>0.4417</td>
      <td>0.9453</td>
      <td>-0.0367</td>
      <td>0.3510</td>
      <td>0.1189</td>
      <td>-0.2433</td>
      <td>0.5781</td>
      <td>0.6747</td>
      <td>-0.5342</td>
      <td>0.4466</td>
      <td>1.1229</td>
      <td>-1.7680</td>
      <td>1.2412</td>
      <td>-2.4495</td>
      <td>-1.7473</td>
      <td>-0.3355</td>
      <td>0.1025</td>
      <td>0.6051</td>
      <td>0.0231</td>
      <td>-0.6265</td>
      <td>0.4791</td>
      <td>-0.1669</td>
      <td>0.0812</td>
      <td>0.0012</td>
      <td>1.1800</td>
      <td>0</td>
    </tr>
    <tr>
      <th>115</th>
      <td>74.0000</td>
      <td>1.0384</td>
      <td>0.1275</td>
      <td>0.1845</td>
      <td>1.1099</td>
      <td>0.4417</td>
      <td>0.9453</td>
      <td>-0.0367</td>
      <td>0.3510</td>
      <td>0.1189</td>
      <td>-0.2433</td>
      <td>0.5781</td>
      <td>0.6747</td>
      <td>-0.5342</td>
      <td>0.4466</td>
      <td>1.1229</td>
      <td>-1.7680</td>
      <td>1.2412</td>
      <td>-2.4495</td>
      <td>-1.7473</td>
      <td>-0.3355</td>
      <td>0.1025</td>
      <td>0.6051</td>
      <td>0.0231</td>
      <td>-0.6265</td>
      <td>0.4791</td>
      <td>-0.1669</td>
      <td>0.0812</td>
      <td>0.0012</td>
      <td>1.1800</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>282987</th>
      <td>171288.0000</td>
      <td>1.9125</td>
      <td>-0.4552</td>
      <td>-1.7507</td>
      <td>0.4543</td>
      <td>2.0891</td>
      <td>4.1600</td>
      <td>-0.8813</td>
      <td>1.0817</td>
      <td>1.0229</td>
      <td>0.0054</td>
      <td>-0.5420</td>
      <td>0.7450</td>
      <td>-0.3752</td>
      <td>-0.0682</td>
      <td>-0.7959</td>
      <td>-0.4979</td>
      <td>-0.1342</td>
      <td>-1.0050</td>
      <td>0.0861</td>
      <td>-0.2076</td>
      <td>-0.5241</td>
      <td>-1.3375</td>
      <td>0.4739</td>
      <td>0.6167</td>
      <td>-0.2835</td>
      <td>-1.0848</td>
      <td>0.0731</td>
      <td>-0.0360</td>
      <td>11.9900</td>
      <td>0</td>
    </tr>
    <tr>
      <th>283483</th>
      <td>171627.0000</td>
      <td>-1.4644</td>
      <td>1.3681</td>
      <td>0.8160</td>
      <td>-0.6013</td>
      <td>-0.6891</td>
      <td>-0.4872</td>
      <td>-0.3038</td>
      <td>0.8850</td>
      <td>0.0541</td>
      <td>-0.8280</td>
      <td>-1.1926</td>
      <td>0.9450</td>
      <td>1.3725</td>
      <td>-0.0365</td>
      <td>-0.2087</td>
      <td>0.3201</td>
      <td>-0.2049</td>
      <td>-0.0248</td>
      <td>-0.4689</td>
      <td>0.0320</td>
      <td>0.2872</td>
      <td>0.9478</td>
      <td>-0.2188</td>
      <td>0.0829</td>
      <td>0.0441</td>
      <td>0.6393</td>
      <td>0.2136</td>
      <td>0.1193</td>
      <td>6.8200</td>
      <td>0</td>
    </tr>
    <tr>
      <th>283485</th>
      <td>171627.0000</td>
      <td>-1.4580</td>
      <td>1.3782</td>
      <td>0.8115</td>
      <td>-0.6038</td>
      <td>-0.7119</td>
      <td>-0.4717</td>
      <td>-0.2825</td>
      <td>0.8807</td>
      <td>0.0528</td>
      <td>-0.8306</td>
      <td>-1.1918</td>
      <td>0.9429</td>
      <td>1.3726</td>
      <td>-0.0380</td>
      <td>-0.2085</td>
      <td>0.3219</td>
      <td>-0.2060</td>
      <td>-0.0252</td>
      <td>-0.4684</td>
      <td>0.0237</td>
      <td>0.2842</td>
      <td>0.9497</td>
      <td>-0.2169</td>
      <td>0.0833</td>
      <td>0.0449</td>
      <td>0.6399</td>
      <td>0.2194</td>
      <td>0.1168</td>
      <td>11.9300</td>
      <td>0</td>
    </tr>
    <tr>
      <th>284191</th>
      <td>172233.0000</td>
      <td>-2.6679</td>
      <td>3.1605</td>
      <td>-3.3560</td>
      <td>1.0078</td>
      <td>-0.3774</td>
      <td>-0.1097</td>
      <td>-0.6672</td>
      <td>2.3097</td>
      <td>-1.6393</td>
      <td>-1.4498</td>
      <td>-0.5089</td>
      <td>0.6000</td>
      <td>-0.6273</td>
      <td>1.0175</td>
      <td>-0.8874</td>
      <td>0.4201</td>
      <td>1.8565</td>
      <td>1.3151</td>
      <td>1.0961</td>
      <td>-0.8217</td>
      <td>0.3915</td>
      <td>0.2665</td>
      <td>-0.0799</td>
      <td>-0.0964</td>
      <td>0.0867</td>
      <td>-0.4511</td>
      <td>-1.1837</td>
      <td>-0.2222</td>
      <td>55.6600</td>
      <td>0</td>
    </tr>
    <tr>
      <th>284193</th>
      <td>172233.0000</td>
      <td>-2.6916</td>
      <td>3.1232</td>
      <td>-3.3394</td>
      <td>1.0170</td>
      <td>-0.2931</td>
      <td>-0.1671</td>
      <td>-0.7459</td>
      <td>2.3256</td>
      <td>-1.6347</td>
      <td>-1.4402</td>
      <td>-0.5119</td>
      <td>0.6079</td>
      <td>-0.6276</td>
      <td>1.0230</td>
      <td>-0.8883</td>
      <td>0.4134</td>
      <td>1.8604</td>
      <td>1.3166</td>
      <td>1.0945</td>
      <td>-0.7910</td>
      <td>0.4026</td>
      <td>0.2597</td>
      <td>-0.0866</td>
      <td>-0.0976</td>
      <td>0.0837</td>
      <td>-0.4536</td>
      <td>-1.2055</td>
      <td>-0.2130</td>
      <td>36.7400</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>1081 rows × 31 columns</p>
</div>



If duplicate exists, it can be removed by using the function: df_card.drop_duplicates()

3. Inconsistent data type 


```python
# datatype checking 
df_card.dtypes
```




    Time      float64
    V1        float64
    V2        float64
    V3        float64
    V4        float64
    V5        float64
    V6        float64
    V7        float64
    V8        float64
    V9        float64
    V10       float64
    V11       float64
    V12       float64
    V13       float64
    V14       float64
    V15       float64
    V16       float64
    V17       float64
    V18       float64
    V19       float64
    V20       float64
    V21       float64
    V22       float64
    V23       float64
    V24       float64
    V25       float64
    V26       float64
    V27       float64
    V28       float64
    Amount    float64
    Class     int64  
    dtype: object



None of the columns have inconsistent datatype.Hence, no conversion is required. 
        If inconsistent data exits, it can be converted by using function: df_data.astype('int32'), 'int32' is an example
