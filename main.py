from pipeline import Pipeline
import pandas as pd


p = Pipeline(pipeline_config='config/kaggle.yaml')
p.preprocess()
res, ids = p.inference()
df = pd.DataFrame(data={
    'id': ids,
    'similarity': res
})
df.to_csv('wrd_kaggle.csv', index=False)