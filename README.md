Factors that are Correlated With Victims' Ages in US Police Shootings
=====================================================================

Data Source
-----------
This dataset was downloaded from [Kaggle](https://www.kaggle.com/). It contains information on individuals who were shot and killed by police between Jan 1 2015 and Jun 14 2020.


It contains victims' names, the date they were shot, how they were killed, whether they were armed (and how they were armed), victims' ages, 
victims' races, the cities and states in which the victims were killed, whether the victims showed any signs of mental illness, the victims' genders, 
whether (and how) the victims were fleeing, the victims' threat levels (were they attacking or not), and whether or not the police were wearing body cameras.


The original dataset can be accessed [here](https://www.kaggle.com/ahsen1330/us-police-shootings).

Motivation
----------
The dataset was compiled from various other Kaggle datasets by the Kaggle user because he wanted to perform analyses surrounding the issue of racism. While
there are many very valid questions surrounding this issue, I wanted to explore this dataset from a different angle.


### The range of the victims' ages is extremely broad: from 6 to 91. 
* What factors are correlated with a victim's age?
* Specifically, which factors are correlated with a victim being younger?
* Are unarmed individuals of certain ages more likely to be killed?
* Is an individual more likely to be killed at a certain age based on location?

Required Libraries
------------------
* python
* pandas
* matplotlib.pyplot
* seaborn
* sklearn
* numpy

Files
-----
* `README.md` - This file
* `LICENSE.md` - Legal information
* the directory `data/` - contains raw data used in this analysis
* `police_shootings.ipynb` - the jupyter notebook that contains my analysis

Results
-------
* The type of weapon a victim has in their possession is 
correlated with their age. 
* Being unarmed is inversely 
correlated with age, so younger people are most likely to be shot while unarmed.

* Age is also inversely correlated with being Native, Black, Hispanic or "Other", 
while it is positively correlated with being White. 

* Victims who are not fleeing are likely to be older.

* The median age of unarmed victims is lower than victims who possess almost any other type of arms.

* The state in which the victim is shot is
correlated with the victim's age. Individuals who are shot in Rhode Island, New York, the District of Columbia 
and Maine are likely to be older.

Next Steps
----------
* It would be informative to compare the distributions of victims' ages in the various states with the distributions
of the ages and races of the inhabitants of those states. Are they similar?


