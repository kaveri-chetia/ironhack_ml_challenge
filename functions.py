import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def model_testing(X_train, X_test, y_train, y_test, model_dict):

    model_types_list = []
    r2_scores = []
    maes = []
    mses = []

    #loop over models
    for model_type_name, model_type in model_dict.items():
        print(f'Model type: {model_type_name}')
        # Create model
        model = model_type

        # Train the model
        model.fit(X_train, y_train)

        # Calculate  R2 Score
        r2_score =  model.score(X_test, y_test)

        # calculate prediction
        pred = model.predict(X_test)

        model_types_list.append(model_type_name)
        r2_scores.append(r2_score)
        maes.append(mean_absolute_error(pred, y_test))
        mses.append(mean_squared_error(pred, y_test, squared=False))

    df_result = pd.DataFrame({'model_type': model_types_list,
                              'r2_score': r2_scores,
                              'MAE': maes,
                              'MSE': mses})
    
    return df_result