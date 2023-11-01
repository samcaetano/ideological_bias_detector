import json
import numpy as np
from statsmodels.stats.contingency_tables import mcnemar


if __name__ == '__main__':
    with open('json/govbr.baseline.bertabaporu-base.predictions.json') as f:
        bertabaporu_predictions = json.load(f)

    with open('json/govbr.baseline.bertimbau-base.predictions.json') as f:
        bertimbau_predictions = json.load(f)


    # Setup for McNemar calculation
    y_true = np.array(bertabaporu_predictions['references']) # This could be any 'references'

    y_pred_bertimbau = np.array(bertimbau_predictions['predictions'])
    y_pred_bertabaporu = np.array(bertabaporu_predictions['predictions'])


    ct = np.array([[0, 0], [0, 0]])
    for idx in range(802):
        # model 1 and model 2 right
        if y_true[idx] == y_pred_bertimbau[idx] and y_true[idx] == y_pred_bertabaporu[idx]:
            ct[0, 0] += 1

        # model 1 right, model 2 wrong
        elif y_true[idx] == y_pred_bertimbau[idx] and y_true[idx] != y_pred_bertabaporu[idx]:
            ct[0, 1] += 1

        # model 2 right, model 1 wrong
        elif y_true[idx] != y_pred_bertimbau[idx] and y_true[idx] == y_pred_bertabaporu[idx]:
            ct[1, 0] += 1

        # model 1 and model 2 wrong
        else:
            ct[1, 1] += 1

    print(mcnemar(ct, exact=False, correction=True))