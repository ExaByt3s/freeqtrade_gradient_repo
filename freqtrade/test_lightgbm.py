import lightgbm
import numpy
from pandas import DataFrame
from load_data import load_data

x_train, y_train, x_test, y_test, column_feature = load_data(return_column_feature=True)
print(x_train.shape, x_test.shape)

x_train = numpy.squeeze(x_train)
x_test = numpy.squeeze(x_test)
print(x_train.shape, x_test.shape)

# model = lightgbm()
# model.fit(x_train, y_train)
#
# print(f'accuracy_train: {model.score(x_train, y_train)}')
# print(f'accuracy_test: {model.score(x_test, y_test)}')

lightgbm_train = lightgbm.Dataset(x_train, y_train)
lightgbm_test = lightgbm.Dataset(x_test, y_test, reference=lightgbm_train)

params = {
    'objective': 'multiclass',
    'num_class': 2,
    # 'num_leaves': 2000,
}

lightgbm_results = {}
model = lightgbm.train(params=params, train_set=lightgbm_train, valid_sets=[lightgbm_train, lightgbm_test],
                       valid_names=['Train', 'Test'], num_boost_round=100, early_stopping_rounds=20,
                       evals_result=lightgbm_results)

# print(lightgbm_results)
# model = lgb.train(lgbm_params, lgb_train, valid_sets=lgb_eval)

y_prediction_probality = model.predict(x_test, num_iteration=model.best_iteration)
y_prediction = numpy.argmax(y_prediction_probality, axis=1)

accuracy = sum(y_test == y_prediction) / len(y_test)
print(accuracy)

# print(model.feature_importance())
importance = DataFrame({'feature': column_feature, 'importance': model.feature_importance()}
                       ).sort_values('importance', ascending=False)
print(importance.to_markdown())
