import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier

if __name__ == "__main__":
    # read data
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    # print data info
    print train.info()
    print test.info()

    # set feature
    select_feature = ['Pclass', 'Sex', 'Age', 'Embarked', 'SibSp', 'Parch', 'Fare']
    x_train = train[select_feature]
    x_test = test[select_feature]
    y_train = train['Survived']
    print x_train['Embarked'].value_counts()
    print x_test['Embarked'].value_counts()

    # fill missing data
    x_train['Embarked'].fillna('S', inplace=True)
    x_test['Embarked'].fillna('S', inplace=True)
    x_train['Age'].fillna(x_train['Age'].mean(), inplace=True)
    x_test['Age'].fillna(x_train['Age'].mean(), inplace=True)
    x_test['Fare'].fillna(x_train['Fare'].mean(), inplace=True)

    # feature to vector
    dict_vec = DictVectorizer(sparse=False)
    x_train = dict_vec.fit_transform(x_train.to_dict(orient='record'))
    x_test = dict_vec.fit_transform(x_test.to_dict(orient='record'))

    rfc = RandomForestClassifier()
    rfc.fit(x_train, y_train)
    rfc_y_predict = rfc.predict(x_test)
    rfc_submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': rfc_y_predict})
    rfc_submission.to_csv('rfc_submission.csv', index=False)
