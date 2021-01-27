import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelBinarizer

if __name__ == '__main__':
    df = pd.read_json("referate-dev.json")

    df = df[['text', 'grade', 'category']]

    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=6, norm='l2', encoding='utf-8', ngram_range=(1, 1),
                            analyzer='word',
                            stop_words=None)
    categories_encoded = LabelBinarizer().fit_transform(df.category)
    # tf_idf_features = tfidf.fit_transform(df.text)

    x_train, x_test, y_train, y_test = train_test_split(df.text, np.array(df.grade))

    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    # Instantiate model with 1000 decision trees
    rf = RandomForestRegressor()
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=3, verbose=2,
                                   random_state=42, n_jobs=-1)
    # Train the model on training data
    clf = make_pipeline(tfidf, rf_random)
    clf.fit(x_train, y_train)

    y_predicted = clf.predict(x_test)

    err = abs(y_predicted - y_test)
    print(round(np.mean(err), 2))

    test_df = pd.read_json("referate-test.json")
    test_df["grade"] = np.nan

    for index, row in test_df.iterrows():
        test_df._set_value(index, 'grade', clf.predict([row.text])[0])

    test_df.to_json("referate-test-completed.json", orient="records")
