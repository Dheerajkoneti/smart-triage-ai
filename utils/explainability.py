def get_feature_importance(model, input_data):

    importances = model.feature_importances_
    features = input_data.columns

    importance_dict = dict(zip(features, importances))

    sorted_features = sorted(
        importance_dict.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return sorted_features[:3]
