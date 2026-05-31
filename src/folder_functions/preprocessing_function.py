import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.impute import KNNImputer
from sklearn.cluster import DBSCAN


def preprocessed_dataset(df):
    df = df.copy()

    # Index and cleanup
    df.set_index('customer_id', inplace=True)

    coordinates = df[['latitude', 'longitude']].copy() # Save coordinates for later use

    df.drop(columns=['Unnamed: 0', 'customer_name', 'longitude', 'latitude'], inplace=True)

    # Gender encoding
    df['customer_gender'] = df['customer_gender'].map({'female': 0, 'male': 1})

    # Birthdate to age
    df['customer_birthdate'] = pd.to_datetime(df['customer_birthdate'])
    df['customer_age'] = ((pd.to_datetime("today") - df['customer_birthdate']).dt.days // 365)
    df.drop(columns=['customer_birthdate'], inplace=True)

    # Time since first transaction
    current_year = pd.to_datetime("today").year
    df['years_since_first_transaction'] = current_year - df['year_first_transaction']
    df.drop(columns=['year_first_transaction'], inplace=True)

    # Promotion variable binary
    df['percentage_of_products_bought_promotion'] = df['percentage_of_products_bought_promotion'].map(lambda x: 1 if x > 0 else 0)

    # Loyalty card: 1 = missing (not loyal), 0 = has card
    df['loyalty_card_number'] = df['loyalty_card_number'].map(lambda x: 1 if pd.isnull(x) else 0)

    # KNN Imputation
    imputer = KNNImputer(n_neighbors=5, weights='distance')
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns, index=df.index)

    # Round categorical floats
    for col in ['kids_home', 'teens_home', 'number_complaints', 'distinct_stores_visited']:
        df[col] = round(df[col], 0)

    # Manual outlier removal based on IQR
    to_drop = [2116, 11695, 5696, 34106, 5501, 21918, 7499, 27580, 18019, 4943,
               38416, 23124, 25785, 33006, 3627, 12411, 30783, 27398, 24821]
    df.drop(index=[i for i in to_drop if i in df.index], inplace=True)

    df = df.drop(columns=['customer_gender', 'percentage_of_products_bought_promotion'])

    # Robust scaling
    scaler = RobustScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)

    # Remove multidimensional outliers using DBSCAN
    dbscan = DBSCAN(eps=2.55, min_samples=20)
    dbscan.fit(df_scaled)
    labels = dbscan.labels_
    df_scaled['cluster'] = labels
    df_scaled = df_scaled[df_scaled['cluster'] != -1].drop(columns=['cluster'])


    return df_scaled, scaler, coordinates

