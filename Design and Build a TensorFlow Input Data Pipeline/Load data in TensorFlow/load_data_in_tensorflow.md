# Chargement de données dans TensorFlow 2.0

Ce notebook présente comment charger des fichiers de différent types dans TensorFlow 2.0.  
Les types de fichiers traités dans le notebook sont les suivants:

1. Tensor
2. Fichier CSV

Pour charger un fichier dans TensorFlow, il faut utiliser l'API `tf.data.Dataset`

## Chargement des bibliothèques utiles


```python
import numpy as np
import pandas as pd 
import tensorflow as tf
from functools import partial


print('TensorFlow vertion: ', tf.version.VERSION )
```

    TensorFlow vertion:  2.1.0



```python
# Améliorer la lecture des valeures numpy
# (precision=3) Limiter à 3 les chiffres après la virgule
# (suppress=True) Supprimer la notation scientifique

np.set_printoptions(precision=3, suppress=True)
```

# Tensor

Charger les données d'un tensor dans TensorFlow avec `.from_tensors`et `.from_tensor_slices`


```python
def show_data(dataset):
    for elem in dataset:
        print(elem.numpy())
    
# 2D Tensor (Rank-2)
t1 = tf.constant([[2, 3], [3, 5]])
# .from_tensorsCréer un dataset contenant seulement un élément
ds1 = tf.data.Dataset.from_tensors(t)

# 2D Tensor (Rank-2)
t2 = tf.constant([[2, 3], [3, 5]])
# .from_tensor_slices créer un dataset contennant autant d'éléments que lignes qui le composent 
ds2 = tf.data.Dataset.from_tensor_slices(t)

print('.from_tensors:')
show_data(ds1)
print()
print('.from_tensor_slice:')
show_data(ds2)
```

    .from_tensors:
    [[2 3]
     [3 5]]
    
    .from_tensor_slice:
    [2 3]
    [3 5]


# Fichier CSV

1. Charger les données dans TensorFlow à partir d'un dataframe pandas.
2. Charger les données d'un fichier csv en utilisant l'API `experimental.make_csv_datase`.

### Analyser les données avant de les charger dans TensorFlow

- Les données utilisées sont issues des logements Parisiens disposant d'un encadrement de loyer.  
Dans cet exemple le but est de prédire le loyer (ref) d'un appartement en fonction du nombre de pièce, époque, le quartier et la zone.

Explorer les données avant de les importer dans TensorFlow.


```python
file_path = './data/logement-encadrement-des-loyers.csv'
df = pd.read_csv(file_path, sep=';' )
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id_zone</th>
      <th>id_quartier</th>
      <th>nom_quartier</th>
      <th>piece</th>
      <th>epoque</th>
      <th>meuble_txt</th>
      <th>ref</th>
      <th>max</th>
      <th>min</th>
      <th>annee</th>
      <th>ville</th>
      <th>code_grand_quartier</th>
      <th>geo_shape</th>
      <th>geo_point_2d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>11</td>
      <td>77</td>
      <td>Belleville</td>
      <td>4</td>
      <td>Avant 1946</td>
      <td>non meublé</td>
      <td>21.4</td>
      <td>25.68</td>
      <td>14.98</td>
      <td>2020</td>
      <td>PARIS</td>
      <td>7512077</td>
      <td>{"type": "Polygon", "coordinates": [[[2.383226...</td>
      <td>48.8715312006,2.38754923985</td>
    </tr>
    <tr>
      <th>1</th>
      <td>13</td>
      <td>75</td>
      <td>Amérique</td>
      <td>3</td>
      <td>1971-1990</td>
      <td>non meublé</td>
      <td>16.7</td>
      <td>20.04</td>
      <td>11.69</td>
      <td>2020</td>
      <td>PARIS</td>
      <td>7511975</td>
      <td>{"type": "Polygon", "coordinates": [[[2.409402...</td>
      <td>48.8816381673,2.39544016662</td>
    </tr>
    <tr>
      <th>2</th>
      <td>13</td>
      <td>74</td>
      <td>Pont-de-Flandre</td>
      <td>2</td>
      <td>1971-1990</td>
      <td>meublé</td>
      <td>20.2</td>
      <td>24.24</td>
      <td>14.14</td>
      <td>2020</td>
      <td>PARIS</td>
      <td>7511974</td>
      <td>{"type": "Polygon", "coordinates": [[[2.384878...</td>
      <td>48.8955557746,2.38477722927</td>
    </tr>
    <tr>
      <th>3</th>
      <td>13</td>
      <td>75</td>
      <td>Amérique</td>
      <td>1</td>
      <td>1971-1990</td>
      <td>meublé</td>
      <td>24.0</td>
      <td>28.80</td>
      <td>16.80</td>
      <td>2020</td>
      <td>PARIS</td>
      <td>7511975</td>
      <td>{"type": "Polygon", "coordinates": [[[2.409402...</td>
      <td>48.8816381673,2.39544016662</td>
    </tr>
    <tr>
      <th>4</th>
      <td>13</td>
      <td>78</td>
      <td>Saint-Fargeau</td>
      <td>1</td>
      <td>Avant 1946</td>
      <td>meublé</td>
      <td>29.4</td>
      <td>35.28</td>
      <td>20.58</td>
      <td>2020</td>
      <td>PARIS</td>
      <td>7512078</td>
      <td>{"type": "Polygon", "coordinates": [[[2.413813...</td>
      <td>48.8710347391,2.40617153015</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2560 entries, 0 to 2559
    Data columns (total 14 columns):
     #   Column               Non-Null Count  Dtype  
    ---  ------               --------------  -----  
     0   id_zone              2560 non-null   int64  
     1   id_quartier          2560 non-null   int64  
     2   nom_quartier         2560 non-null   object 
     3   piece                2560 non-null   int64  
     4   epoque               2560 non-null   object 
     5   meuble_txt           2560 non-null   object 
     6   ref                  2560 non-null   float64
     7   max                  2560 non-null   float64
     8   min                  2560 non-null   float64
     9   annee                2560 non-null   int64  
     10  ville                2560 non-null   object 
     11  code_grand_quartier  2560 non-null   int64  
     12  geo_shape            2560 non-null   object 
     13  geo_point_2d         2560 non-null   object 
    dtypes: float64(3), int64(5), object(6)
    memory usage: 280.1+ KB



```python
df.describe().T
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>id_zone</th>
      <td>2560.0</td>
      <td>6.662500e+00</td>
      <td>4.225585</td>
      <td>1.00</td>
      <td>3.00</td>
      <td>5.00</td>
      <td>11.00</td>
      <td>14.00</td>
    </tr>
    <tr>
      <th>id_quartier</th>
      <td>2560.0</td>
      <td>4.050000e+01</td>
      <td>23.096718</td>
      <td>1.00</td>
      <td>20.75</td>
      <td>40.50</td>
      <td>60.25</td>
      <td>80.00</td>
    </tr>
    <tr>
      <th>piece</th>
      <td>2560.0</td>
      <td>2.500000e+00</td>
      <td>1.118252</td>
      <td>1.00</td>
      <td>1.75</td>
      <td>2.50</td>
      <td>3.25</td>
      <td>4.00</td>
    </tr>
    <tr>
      <th>ref</th>
      <td>2560.0</td>
      <td>2.572723e+01</td>
      <td>4.181951</td>
      <td>14.60</td>
      <td>22.90</td>
      <td>25.30</td>
      <td>28.30</td>
      <td>39.60</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2560.0</td>
      <td>3.087267e+01</td>
      <td>5.018341</td>
      <td>17.52</td>
      <td>27.48</td>
      <td>30.36</td>
      <td>33.96</td>
      <td>47.52</td>
    </tr>
    <tr>
      <th>min</th>
      <td>2560.0</td>
      <td>1.800906e+01</td>
      <td>2.927365</td>
      <td>10.22</td>
      <td>16.03</td>
      <td>17.71</td>
      <td>19.81</td>
      <td>27.72</td>
    </tr>
    <tr>
      <th>annee</th>
      <td>2560.0</td>
      <td>2.020000e+03</td>
      <td>0.000000</td>
      <td>2020.00</td>
      <td>2020.00</td>
      <td>2020.00</td>
      <td>2020.00</td>
      <td>2020.00</td>
    </tr>
    <tr>
      <th>code_grand_quartier</th>
      <td>2560.0</td>
      <td>7.511090e+06</td>
      <td>599.811459</td>
      <td>7510101.00</td>
      <td>7510595.75</td>
      <td>7511090.50</td>
      <td>7511585.25</td>
      <td>7512080.00</td>
    </tr>
  </tbody>
</table>
</div>



## Créer trois jeux de données
1. Un jeux de données pour l'entrainement du model (80% des 90% de l'ensemble des données)
2. Un jeux de données pour l'évaluation du model (20% des 90% de l'ensemble des données
3. Un jeux de données pour réaliser des testes (10% de l'ensemble des donnée du fichier)


```python
def df_row_to_split(df, frac):
    '''Cette fonction permet de déterminer le nombre de ligne du dataframe à retourner en fonc'''
    percent = frac * 100
    return round(df_sample.shape[0] * percent / 100)

# Mélanger les données du dataframe
df_sample = df.sample(frac=1, random_state=21).reset_index(drop=False)

# Prendre environ 90% des données pour l'entrainement et l'évaluation du model
row_nb = df_row_to_split(df, 0.9)
train_eval_data = df_sample[:row_nb].drop(['index'], axis=1)

# Prendre environ 10% des données pour tester du model sur de nouvelle données
test_data = df_sample[row_nb:].drop(['index'], axis=1)
test_data.to_csv('./data/test.csv', index=False)

# Prendre 80% des données du dataframe train_eval pour l'entrainement du model
row_nb = df_row_to_split(train_eval_data, 0.8)
train_data = train_eval_data[:row_nb]
train_data.to_csv('./data/train.csv', index=False)

# Prendre 20% des données du dataframe train_eval pour l'évaluation du model
eval_data = train_eval_data[row_nb:]
eval_data.to_csv('./data/eval.csv', index=False)

```

### Pré-traitement des données

Traiter les données du dataframe avant de la charger dans TensorFlow.

1. Les valeurs de la colonne `epoque` ne sont pas de type continue, elles sont de type string.  
Les données de cette colonne doivent être transformé pour pouvoir être utiliser.

2. Supprimer les colonnes non utilisées


```python
def df_processed(df, features, label):
    '''Traitement des données avant le chargement dans tensorFlow'''
    
    df_processed = df.copy()
    df_processed.columns
    
    # Listes des noms des colonnes
    features_label = features + label
    # Supprimer les colonnes non utilisées
    col_to_remove = [col_name for col_name in df_processed.columns.tolist() if col_name not in features_label]
    
    return df_processed.drop(col_to_remove, axis=1)

def one_hot_encoding(df, col_names):
    '''Cette fonction permet de traiter les colonnes avec des données catégoriel
    en utilisant la methode de one-hot encoding'''
    
    for col_name in col_names:
        df[col_name] = pd.Categorical(df[col_name])
        df[col_name] = df[col_name].cat.codes
        
    return df
```


```python
# Créer les dataframe pandas 

df_train = pd.read_csv('./data/train.csv', sep=',')
df_eval = pd.read_csv('./data/eval.csv', sep=',')
df_test = pd.read_csv('./data/test.csv', sep=',')
```


```python
label_col_name = ['ref']
features_col_name = ['piece', 'epoque', 'id_zone', 'id_quartier'] 

# Pré-traitement du dataframe avant le chargement dans TensforFlow
df_train_processed = df_processed(df_train, features_col_name, label_col_name)
df_eval_processed = df_processed(df_eval, features_col_name, label_col_name)

# Utiliser la methode du 'One Hote Encoding' pour traiter les données de type catégoriel
df_train_processed = one_hot_encoding(df_train_processed, ['epoque'])
df_eval_processed.epoque = one_hot_encoding(df_eval_processed, ['epoque'])

df_train_processed.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id_zone</th>
      <th>id_quartier</th>
      <th>piece</th>
      <th>epoque</th>
      <th>ref</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>9</td>
      <td>70</td>
      <td>1</td>
      <td>1</td>
      <td>24.2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10</td>
      <td>44</td>
      <td>2</td>
      <td>2</td>
      <td>28.0</td>
    </tr>
  </tbody>
</table>
</div>



### Charger les dataframes pandas dans TensFlow


```python
def show_data(dataset, nb_row):
    '''Cette fonction permet d'afficher les exemples d'un tensor'''
    
    for feat, label in dataset.take(nb_row):
        print('Features: {}, Label: {}'.format(feat, label))
```


```python
# Valeures à prédire 
label_train = df_train_processed.pop(label_col_name[0])
label_eval = df_eval_processed.pop(label_col_name[0])

# Charger les données dans tensorFlow avec tf.data.Dataset.from_tensor_slices
train_dataset = tf.data.Dataset.from_tensor_slices((df_train_processed.values, label_train.values))
eval_dataset = tf.data.Dataset.from_tensor_slices((df_eval_processed.values, label_eval.values))

show_data(train_dataset, 1)
```

    Features: [ 9 70  1  1], Label: 24.2


### Mélanger les données données et créer des minis batch

Si les données non pas déjà été mélangé:
- `dataset.shuffle(len(df), seed=(21)).batch(nb_of_exemple)`

Dans notre cas les données ont déjà été mélangé précédemment.


```python
# Les minis batch sont volontairement petit pour une meilleure lisibilité des exemples
train_data_set = train_dataset.batch(2)
eval_data_set = eval_dataset.batch(2)
```

Le paramètre `seed` permet de garder les données mélanger dans le même ordre et ça, peu importe le nombre de fois qu' est exécuté le code.

Les ` mini batch` permettent de générer le calcul de la fonction `loss`, de calculer les `gradients` sur un ensemble d'exemples et non pas un exemple à la fois, ce qui permet d'accélérer l'entrainement et de tirer un meilleur parti du GPU qui est plus efficient pour réaliser des calculs matriciels.


```python
show_data(train_data_set, 2)
```

    Features: [[ 9 70  1  1]
     [10 44  2  2]], Label: [24.2 28. ]
    Features: [[14 76  4  3]
     [13 73  1  1]], Label: [20.1 24. ]


# Charger les données d'un fichier csv en utilisant l'API `experimental.make_csv_datase`.

Si le besoin est d'importer un ensemble important de fichiers, utiliser la fonction `tf.data.experimental.make_csv_dataset` 


```python
def get_dataset(file_path, **kwargs):
    '''Cette fonction permet de charger un fichier CSV ou plusieurs fichiers dans un répertoire'''
    
    dataset = tf.data.experimental.make_csv_dataset(
        file_pattern=file_path,
        batch_size=3,
        na_value="?",
        ignore_errors=True, 
        num_epochs=1,
        **kwargs)
    
    return dataset

def show_data(dataset):
    '''Cette fonction permet d'afficher les données contenue dans les mini batch'''
    
    for batch, label in dataset.take(1):
        for key, value in batch.items():
            print("{:20s}: {}".format(key, value.numpy()))
        print()
        print('{:20s}: {}'.format('Labels', label.numpy()))
```

### Importer les fichiers csv dans TensorFlow


```python
train_file = './data/train.csv'
eval_file = './data/eval.csv'

# Liste des colonnes à sélectioner dans le fichier
SELECT_COLUMNS = ['id_zone','id_quartier', 'piece', 'epoque', 'ref']

# Attribuer un format de données à chaque colonne (Optionel)
DEFAULTS = [tf.int32, tf.int32, tf.int32, tf.string, tf.float32]

raw_train_data = get_dataset(train_file,
                             label_name='ref',
                             select_columns=SELECT_COLUMNS,
                             column_defaults=DEFAULTS)

raw_eval_data = get_dataset(eval_file,
                            label_name='ref',
                            select_columns=SELECT_COLUMNS,
                            column_defaults=DEFAULTS)
```


```python
show_data(raw_train_data)
```

    id_zone             : [14 13  5]
    id_quartier         : [79 50 53]
    piece               : [1 1 1]
    epoque              : [b'Apres 1990' b'Apres 1990' b'Apres 1990']
    
    Labels              : [29.  23.1 31.5]


### Pré-traitement des données numerique


```python
class PackNumericFeatures(object):
    '''Cette class permet de créer un vecteur avec toutes les caratéristiques'''
    def __init__(self, names):
        self.names = names

    def __call__(self, features, labels):
        numeric_features = [features.pop(name) for name in self.names]
        # Mettre les données numéric au format de type float32
        numeric_features = [tf.cast(feat, tf.float32) for feat in numeric_features]
        # Empiler les données dans un vecteur
        numeric_features = tf.stack(numeric_features, axis=-1)
        # Ajouter le vecteur aux caractéristiques
        features['numeric'] = numeric_features
        
        return features, labels
```


```python
NUMERIC_FEATURES = ['piece', 'id_zone', 'id_quartier']

# Créer un vecteur contenant les valeurs numéric qui sera ingéré par le model
packed_train_ds = raw_train_data.map(PackNumericFeatures(NUMERIC_FEATURES))
```


```python
show_data(packed_train_ds)
```

    epoque              : [b'Apres 1990' b'1946-1970' b'1946-1970']
    numeric             : [[ 4.  9. 46.]
     [ 2.  1. 23.]
     [ 2. 11. 39.]]
    
    Labels              : [20.  30.1 25. ]



```python
example_batch, labels_batch = next(iter(packed_train_ds))
```

### Normaliser les caratéristiques

Les données de type continue doivent toujours être normalisé.   
Normaliser les données permet d'accélérer la recherche du minima de la fonction de perte durant la descente des gradients


```python
def standar_scaler(data, mean, std):
    '''
    Cette fonction permet de normaliser les variables (data) pour qu'elles aient une moyenne nulle et une variance
    égale à 1. Pour une variable, cela correspond à retrancher à chaque observation la moyenne (mean) de la variable et à diviser chaque observation
    par l'écart-type (std).
    '''
    
    return (data-mean) / std

def mix_max_scaler(data, min_, max_):
    '''
    Cette fonction permet de normaliser une variable pour qu'elles evoluent en 0 et 1.
    Pratique si besoin de probabilité.
    '''
    return (data-min_) / (max_ - min_)
```


```python
desc = pd.read_csv(train_file, sep=',')[NUMERIC_FEATURES].describe()
desc
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>piece</th>
      <th>id_zone</th>
      <th>id_quartier</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2048.000000</td>
      <td>2048.000000</td>
      <td>2048.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2.492188</td>
      <td>6.644043</td>
      <td>40.350586</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.117406</td>
      <td>4.217204</td>
      <td>23.005896</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>20.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2.000000</td>
      <td>5.000000</td>
      <td>40.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>3.000000</td>
      <td>11.000000</td>
      <td>60.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>4.000000</td>
      <td>14.000000</td>
      <td>80.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
MEAN = np.array(desc.T['mean'])
STD = np.array(desc.T['mean'])
MAX = np.array(desc.T['max'])
MIN = np.array(desc.T['min'])

print('Mean            : ', MEAN)
print('Ecart-Type (std): ', STD)
print('Max             : ', MAX)
print('Min             : ', MIN)
```

    Mean            :  [ 2.4921875   6.64404297 40.35058594]
    Ecart-Type (std):  [ 2.4921875   6.64404297 40.35058594]
    Max             :  [ 4. 14. 80.]
    Min             :  [1. 1. 1.]



```python
from functools import partial

# Lier les valeures mean et std aux fonctions de normalisation
standar_scaler = partial(standar_scaler, mean=MEAN, std=STD)
mix_max_scaler = partial(mix_max_scaler, min_=MIN, max_=MAX)

numeric_column = tf.feature_column.numeric_column('numeric',
                                                  normalizer_fn=standar_scaler,
                                                  shape=[len(NUMERIC_FEATURES)])
numeric_column = [numeric_column]
numeric_column
```




    [NumericColumn(key='numeric', shape=(3,), default_value=None, dtype=tf.float32, normalizer_fn=functools.partial(<function standar_scaler at 0x13a6fc560>, mean=array([ 2.4921875 ,  6.64404297, 40.35058594]), std=array([ 2.4921875 ,  6.64404297, 40.35058594])))]




```python
example_batch['numeric']
```




    <tf.Tensor: shape=(3, 3), dtype=float32, numpy=
    array([[ 1.,  3., 32.],
           [ 4., 11., 77.],
           [ 2., 10., 44.]], dtype=float32)>




```python
# Normaliser les caratéristiques dans tous les mini batch
numeric_layer = tf.keras.layers.DenseFeatures(numeric_column)
numeric_layer(example_batch).numpy()
```




    array([[-0.59874606, -0.5484677 , -0.2069508 ],
           [ 0.6050157 ,  0.6556184 ,  0.90827465],
           [-0.19749217,  0.50510764,  0.09044265]], dtype=float32)



### Traiter les données catégoriel

Certaines des colonnes du fichier contiennent des données catégoriel de type string,  
Réaliser un `one-hot encoding`  en utilsant l'API `tf.feature_column` et `indicator_column`


```python
CATEGORIES = {'epoque': ['Avant 1946', '1971-1990', 'Apres 1990', '1946-1970']}
```


```python
categorical_column = []

for feature, vocab in CATEGORIES.items():
    cat_col = (tf.feature_column.
               categorical_column_with_vocabulary_list(key=feature, vocabulary_list=vocab))
    
    categorical_column.append(tf.feature_column.indicator_column(cat_col))
    
categorical_column
```




    [IndicatorColumn(categorical_column=VocabularyListCategoricalColumn(key='epoque', vocabulary_list=('Avant 1946', '1971-1990', 'Apres 1990', '1946-1970'), dtype=tf.string, default_value=-1, num_oov_buckets=0))]




```python
categorial_layer = tf.keras.layers.DenseFeatures(categorical_column)
print(categorial_layer(example_batch).numpy())
```

    [[1. 0. 0. 0.]
     [0. 1. 0. 0.]
     [0. 0. 1. 0.]]


### Combiner toutes les couches traitées


```python
preprocessing_layer = tf.keras.layers.DenseFeatures(categorical_column+numeric_column)
print(preprocessing_layer(example_batch).numpy()[0])
```

    [ 1.          0.          0.          0.         -0.59874606 -0.5484677
     -0.2069508 ]


# Fin

L'étape suivante serait d'utiliser `tf.keras.Sequential` avec les données de `preprocessing_layer`.       
Mais ce n'est pas l'objectif de ce notebook

# Resources 
1. Charger les données text - lien: https://www.tensorflow.org/tutorials/load_data/text
2. TF.text - lien:  https://www.tensorflow.org/tutorials/tensorflow_text/intro
3. Charger des images - https://www.tensorflow.org/tutorials/load_data/images
4. Lire les données d'un dataframe pandas - https://www.tensorflow.org/tutorials/load_data/pandas_dataframe   
