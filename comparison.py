import torch
import numpy
from sklearn.ensemble import RandomForestClassifier
from binary_classification import load_data, train, predict, accuracy


'''
As my secondary classifier I chose to use a random forest classifier. 
This model builds a series of decision trees from the dataset considering
different feature combinations trees before returning the classification 
reached by the majority of trees. I chose it because it is able to consider 
multiple features at a time and take into account how they may combine while
keeping the theory behind the model simple an concise. 
'''

if __name__ == "__main__":
    X_train_norm, X_test_norm, y_train, y_test, feature_names = load_data()
    forest = RandomForestClassifier(n_estimators=1000, max_depth=2, random_state=0)
    forest.fit(X_train_norm, y_train)

    predictions = forest.predict(X_test_norm)
    forest_accuracy = accuracy(y_test, torch.tensor(predictions))

    print(f'model accuracy: {forest_accuracy}%')


    w, b, _ = train(X_train_norm, y_train, alpha=0.01, n_epochs=100, verbose=False)
        
    test_pred = predict(X_test_norm, w, b)
    test_acc = accuracy(y_test, test_pred)
    print(f'from scratch accuracy: {test_acc}')

    '''
    The random forest model performed decently with a 96% test accuracy. 
    However, the froms scratch binary classification network performed better with 99% test accuracy
    '''


