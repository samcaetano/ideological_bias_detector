from models.hybrid_model import NeuralModel
from models.bert_embeddings import Embeddings_builder

train_comple_features, test_comple_features = None, None

kwargs = {
    "corpus": "govbr",
    "num_samples": 4010,
    "mode": "baseline.bert",
    "num_conv_layers": 5,
    "num_comple_features": 16023,
    "num_classes": 2,
}

embedding = Embeddings_builder(kwargs['corpus'])

train = embedding.load_text(f"govbr.train.csv")
test = embedding.load_text(f"govbr.test.csv")
print("Dataset loaded")

# train_comple_features = embedding.load_sngram('byarticles.sngram.train.csv')
# test_comple_features = embedding.load_sngram('byarticles.sngram.test.csv')


neural_model = NeuralModel(
    **kwargs,
    builder=embedding,
    comple_features_train=train_comple_features,
    comple_features_test=test_comple_features,
)

# neural_model.train(train)
neural_model.predict(test, 802)