import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# 1. Carregar CSV
df = pd.read_csv("dados.csv")

# 2. Remover dados sensíveis
df = df.drop(columns=["CPF", "RG", "CNH", "PASSAPORTE", "TELEFONE", "END_RESIDENCIAL"])

# 3. Generalizar dados
df["IDADE"] = df["IDADE"].apply(lambda x: min(max(int(x), 0), 100))
df["FAIXA_ETARIA"] = pd.cut(df["IDADE"], bins=[0,18,25,35,45,60,100],
                            labels=["0-18","19-25","26-35","36-45","46-60","60+"])

# 4. Pré-processamento
features = ["IDADE", "SEXO", "OCUPACAO"]
categorical = ["SEXO", "OCUPACAO"]
numeric = ["IDADE"]

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical),
        ('num', StandardScaler(), numeric)
    ]
)

# 5. Modelo de clusterização
model = Pipeline(steps=[('preprocess', preprocessor),
                        ('cluster', KMeans(n_clusters=5, random_state=42))])

df["CLUSTER"] = model.fit_predict(df[features])

# 6. Estatísticas agregadas
clusters = df.groupby("CLUSTER").agg({
    "IDADE": ["mean", "min", "max"],
    "SEXO": lambda x: x.mode()[0],
    "OCUPACAO": lambda x: x.mode()[0],
    "NOME": "count"
}).reset_index()

clusters.columns = ["CLUSTER", "IDADE_MEDIA", "IDADE_MIN", "IDADE_MAX", "SEXO_MAIS_COMUM", "OCUPACAO_COMUM", "TOTAL_PESSOAS"]

print(clusters)
