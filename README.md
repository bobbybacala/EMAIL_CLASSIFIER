An Email Classifier System, that could categorize mails into multiple classes, trained the classifier on classes like Finance, Operations, HR, Advertisement, Customer Support, entirely based on the dataset, did some EDA on custom made dataset, and used classifiers to predict the class of the incoming mail.

- Libraries used are NumPy, Pandas, NLTK(natural language toolkit) for tokenization, Matplotlib and Seaborn for some Exploratory data analysis.

- Used LabelEncoder Library for Encoding textual data into numerical form.

- Data Cleaning was done by removing duplicates and null values.

- Data Preprocessing is the most important step, in which the actual mail is reduced to a small text which is not so easy to read by humans.

- To train the classifier, we first need the vectorize the textual mail into sparse matrix(a matrix where most values are set to 0) this can be done by 2 ways, CountVectorizer and TFIDF(Term frequency Inverse Document frequency) Vectorizer

- Train the classifier using fit() method, and compare with other classifiers for scope of improvement.