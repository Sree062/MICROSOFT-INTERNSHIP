//Data Pre-Processing
In [ ]:
import pandas as p
import numpy as n
In [ ]:
import warnings
warnings.filterwarnings(&#39;ignore&#39;)
In [ ]:
data=p.read_csv(&#39;kidney.csv&#39;)
In [ ]:

data.head()
In [ ]:
data.shape
In [ ]:
df=data.dropna()
In [ ]:
df.shape
In [ ]:
df.tail()
In [ ]:
df.isnull().sum()
In [ ]:
df.describe()
In [ ]:
df.columns
In [ ]:
df.Bp.unique()
In [ ]:
df.Hemo.unique()
In [ ]:
df.Class.unique()
In [ ]:
df.info()
In [ ]:

df.duplicated()
In [ ]:
sum(df.duplicated())
In [ ]:
df.corr()
In [ ]:
df[&#39;Bp&#39;].value_counts()
In [ ]:
df[&#39;Class&#39;].value_counts()
In [ ]:
df.columns
In [ ]:
print(&quot;Minimum value of Hemoglobin Level is:&quot;, df.Hemo.min())
print(&quot;Maximum value of Hemoglobin Level is:&quot;, df.Hemo.max())
In [ ]:
print(&quot;Minimum value of histogram_mean is:&quot;, df.Bp.min())
print(&quot;Maximum value of histogram_mean is:&quot;, df.Bp.max())
In [ ]:
p.crosstab(df.Class,df.Bp)
In [ ]:
df.head()
In [ ]:
df.tail()
In [ ]:

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
variable = [&#39;Bp&#39;, &#39;Sg&#39;, &#39;Al&#39;, &#39;Su&#39;, &#39;Rbc&#39;, &#39;Bu&#39;, &#39;Sc&#39;, &#39;Sod&#39;, &#39;Pot&#39;, &#39;Hemo&#39;, &#39;Wbcc&#39;,
&#39;Rbcc&#39;, &#39;Htn&#39;, &#39;Class&#39;]
for i in variable:
df[i] = le.fit_transform(df[i]).astype(int)
In [ ]:
df.head()
In [ ]:
df.tail()
Module - 2
Visualization
In [ ]:
import pandas as p
import numpy as n
import seaborn as s
import matplotlib.pyplot as plt
In [ ]:
import warnings
warnings.filterwarnings(&#39;ignore&#39;)
In [ ]:
data=p.read_csv(&#39;kidney.csv&#39;)
In [ ]:
df=data.dropna()
In [ ]:

df.columns
In [ ]:
#Histogram for histogram_max &amp; histogram_min
df[&#39;Hemo&#39;].hist(figsize=(5,5), color=&#39;m&#39;, alpha=1)
plt.xlabel(&#39;Hemoglobin&#39;)
plt.ylabel(&#39;Count Value&#39;)
plt.title(&#39;Hemoglobin Count&#39;)
In [ ]:
#Histogram for histogram_max &amp; histogram_min
df[&#39;Sg&#39;].hist(figsize=(5,5), color=&#39;c&#39;, alpha=1)
plt.xlabel(&#39;Specific Gravity&#39;)
plt.ylabel(&#39;Count Value&#39;)
plt.title(&#39;Histogram for Specific Gravity&#39;)
In [ ]:
plt.boxplot(df[&#39;Hemo&#39;])
plt.show()
In [ ]:
import seaborn as s
s.boxplot(df[&#39;Sg&#39;], color=&#39;m&#39;)
In [ ]:
fig, ax = plt.subplots(figsize=(16,8))
ax.scatter(df[&#39;Wbcc&#39;],df[&#39;Rbcc&#39;])
ax.set_xlabel(&#39;White Blood Cells Count&#39;)
ax.set_ylabel(&#39;Red Blood Cells Count&#39;)
plt.show()
In [ ]:

#Propagation by variable
def PropByVar(df, variable):
dataframe_pie = df[variable].value_counts()
ax = dataframe_pie.plot.pie(figsize=(10,10), autopct=&#39;%1.2f%%&#39;, fontsize =
12)
ax.set_title(variable + &#39; \n&#39;, fontsize = 15)
return n.round(dataframe_pie/df.shape[0]*100,2)

PropByVar(df, &#39;Sg&#39;)
In [ ]:
#Propagation by variable
def PropByVar(df, variable):
dataframe_pie = df[variable].value_counts()
ax = dataframe_pie.plot.pie(figsize=(10,10), autopct=&#39;%1.2f%%&#39;, fontsize =
12)
ax.set_title(variable + &#39; \n&#39;, fontsize = 15)
return n.round(dataframe_pie/df.shape[0]*100,2)

PropByVar(df, &#39;Class&#39;)
In [ ]:
fig, ax = plt.subplots(figsize=(15,7))
s.heatmap(df.corr(), ax=ax, annot=True)
In [ ]:
plt.figure(figsize=(15,6))
s.countplot(&#39;Bp&#39;,hue=&#39;Class&#39;,data=df)
In [ ]:
s.histplot(df[&#39;Hemo&#39;])

Module - 3
Random Forest Algorithm
In [ ]:
#importing Library Packages
import pandas as p
import numpy as n
import matplotlib.pyplot as plt
import seaborn as s
In [ ]:
import warnings
warnings.filterwarnings(&#39;ignore&#39;)
In [ ]:
data=p.read_csv(&#39;kidney.csv&#39;)
In [ ]:
df=data.dropna()
In [ ]:
df.columns
In [ ]:
#preprocessing, split test and dataset, split response variable
X = df.drop(labels=&#39;Class&#39;, axis=1)
#Response variable
y = df.loc[:,&#39;Class&#39;]
In [ ]:

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,
random_state=1, stratify=y)
print(&quot;Number of training dataset: &quot;, len(X_train))
print(&quot;Number of test dataset: &quot;, len(X_test))
print(&quot;Total number of dataset: &quot;, len(X_train)+len(X_test))
In [ ]:
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
In [ ]:
from sklearn.ensemble import RandomForestClassifier

rfc= RandomForestClassifier()
rfc.fit(X_train,y_train)
predictRF = rfc.predict(X_test)

print(&quot;&quot;)
print(&#39;Classification report of Random Forest Results:&#39;)
print(&quot;&quot;)
print(classification_report(y_test,predictRF))

print(&quot;&quot;)
cm1=confusion_matrix(y_test,predictRF)
print(&#39;Confusion Matrix result of Random Forest Classifier is:\n&#39;,cm1)
print(&quot;&quot;)

accuracy = cross_val_score(rfc, X, y, scoring=&#39;accuracy&#39;)
print(&#39;Cross validation test results of accuracy:&#39;)
print(accuracy)
#get the mean of each fold
print(&quot;&quot;)
print(&quot;Accuracy result of Random Forest Classifier is:&quot;,accuracy.mean() * 100)
RF=accuracy.mean() * 100
In [ ]:
def acc_bar():
import matplotlib.pyplot as plt
data=[RF]
alg=&quot;Random Forest Classifier&quot;
plt.figure(figsize=(5,6))
b=plt.bar(alg,data,color=(&quot;m&quot;))
plt.title(&quot;Accuracy comparison of Chronic Renal Disease&quot;,fontsize=15)
plt.legend(b,data,fontsize=9)
acc_bar()
In [ ]:
DF = p.DataFrame()
DF[&quot;y_test&quot;] = y_test
DF[&quot;predicted&quot;] = predictRF
DF.reset_index(inplace=True)
plt.figure(figsize=(20, 5))
plt.plot(DF[&quot;predicted&quot;][:100], marker=&#39;x&#39;, linestyle=&#39;dashed&#39;, color=&#39;red&#39;)
plt.plot(DF[&quot;y_test&quot;][:100], marker=&#39;o&#39;, linestyle=&#39;dashed&#39;, color=&#39;green&#39;)
plt.show()
Creating pkl File

In [ ]:
import joblib
joblib.dump(rfc, &#39;rf.pkl&#39;)
