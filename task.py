import numpy as np
import pandas as pd
# f = open("dataset/referate-dev.json")
# dataset_json = json.load(f)
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
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

    # Instantiate model with 1000 decision trees
    rf = RandomForestRegressor(n_estimators=1000, random_state=42)
    # Train the model on training data
    clf = make_pipeline(tfidf, rf)
    clf.fit(x_train, y_train)

    # for i in range(10):
    #     predicted = rf.predict(x_test[i])
    #     print('Predicted {}, Real {}'.format(predicted, y_test[i]))

    y_predicted = clf.predict(x_test)

    err = abs(y_predicted - y_test)
    print(round(np.mean(err), 2))

    test_df = pd.read_json("referate-test.json")
    test_df["grade"] = np.nan

    for index, row in test_df.iterrows():
        test_df._set_value(index, 'grade', clf.predict([row.text])[0])

    test_df.to_json("referate-test-completed.json", orient="records")
